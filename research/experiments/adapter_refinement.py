"""Prepare auditable reductions in ordinary LoRA strength, without training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

try:
    from experiments.confirmation import adapter_hashes, digest
except ModuleNotFoundError:
    from confirmation import adapter_hashes, digest

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SCALES = {'dpo-32-scale050': .5, 'dpo-32-scale075': .75}


def timestamp(value):
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if date.tzinfo is None:
        raise ValueError('Registration timestamp needs a timezone')
    return date.timestamp()


def registered_specs(protocol):
    names = protocol.get('additional_refinement_candidates', [])
    specs = protocol.get('additional_refinement_specs', {})
    if not names and not specs:
        return {}
    if len(names) != len(SCALES) or set(names) != set(SCALES) or set(specs) != set(SCALES):
        raise ValueError('Additional refinement registration must contain exactly the two bounded variants')
    for name, spec in specs.items():
        if spec.get('parent_candidate') != 'dpo-32' or spec.get('strength_multiplier') != SCALES[name]:
            raise ValueError('Unexpected refinement parent or strength')
        timestamp(spec['registered_at'])
    return specs


def verify_registration(root, name, registration, parent_registration, spec):
    root = Path(root).resolve()
    def path(value):
        result = (root / value).resolve()
        if not result.is_relative_to(root):
            raise ValueError('Refinement path escapes study workspace')
        return result
    if name not in SCALES or spec['strength_multiplier'] != SCALES[name]:
        raise ValueError('Unknown refinement')
    if registration.get('parent_candidate') != 'dpo-32':
        raise ValueError('Registry parent differs from registered refinement')
    if registration.get('refinement_manifest') != spec['manifest']:
        raise ValueError('Registry and protocol name different refinement manifests')
    manifest_path = path(spec['manifest'])
    if digest(manifest_path) != spec['manifest_sha256']:
        raise ValueError('Refinement manifest differs from registration')
    manifest = json.loads(manifest_path.read_text())
    if manifest['candidate'] != name or manifest['parent_candidate'] != 'dpo-32':
        raise ValueError('Refinement manifest identity differs')
    if manifest['strength_multiplier'] != spec['strength_multiplier']:
        raise ValueError('Refinement strength differs from registration')
    if timestamp(manifest['prepared_at']) > timestamp(spec['registered_at']):
        raise ValueError('Preparation occurred after claimed registration')
    parent = root / 'lab/experiments/data/study/adapters/dpo-32'
    if path(manifest['parent_source']) != parent:
        raise ValueError('Manifest parent is not the frozen staged DPO-32 adapter')
    archived = adapter_hashes(path(parent_registration['source']))
    staged_parent = adapter_hashes(parent)
    if set(archived) - set(staged_parent) - {'trainer_state.pt'} or set(staged_parent) - set(archived):
        raise ValueError('Parent file sets differ beyond omitted optimizer state')
    if any(archived[k] != value for k, value in staged_parent.items()):
        raise ValueError('Parent inference files differ from their training archive')
    for field, file in [('checkpoint_sha256', 'adapter_model.safetensors'),
                        ('config_sha256', 'adapter_config.json')]:
        if staged_parent[file] != parent_registration.get(field):
            raise ValueError('Parent differs from the original checkpoint registry')
        if manifest['adapter_files_sha256'][file] != registration.get(field):
            raise ValueError('Refinement differs from the checkpoint registry')
    verify(parent, path(registration['source']), manifest)
    verify(parent, root / 'lab/experiments/data/study/adapters' / name, manifest)
    return {'parent_candidate': 'dpo-32', 'strength_multiplier': spec['strength_multiplier'],
            'registered_at': spec['registered_at'], 'manifest_sha256': spec['manifest_sha256'],
            'change': manifest['change']}


def scaled_config(config, scale):
    if type(scale) not in (int, float) or not 0 < scale < 1:
        raise ValueError('Strength must be strictly between zero and one')
    if config.get('peft_type') != 'LORA' or config.get('bias') != 'none':
        raise ValueError('Only ordinary bias-free LoRA is supported')
    if any(config.get(k) for k in ('use_dora', 'use_rslora', 'lora_bias', 'modules_to_save',
                                   'rank_pattern', 'alpha_pattern')):
        raise ValueError('Unsupported LoRA variant or additional trainable parameters')
    if config.get('base_model_name_or_path') != MODEL or config.get('revision') != REVISION:
        raise ValueError('Starting model differs from the pinned study model')
    alpha, rank = config.get('lora_alpha'), config.get('r')
    if type(alpha) is not int or type(rank) is not int or min(alpha, rank) <= 0:
        raise ValueError('Expected positive integer alpha and rank')
    scaled = alpha * scale
    if not float(scaled).is_integer() or scaled <= 0:
        raise ValueError('Refinement must preserve an integer lora_alpha')
    return dict(config, lora_alpha=int(scaled))


def verify(parent, candidate, manifest):
    parent, candidate = Path(parent), Path(candidate)
    original = adapter_hashes(parent)
    actual = adapter_hashes(candidate)
    if original != manifest['parent_files_sha256']:
        raise ValueError('Parent adapter changed after preparation')
    if actual != manifest['adapter_files_sha256']:
        raise ValueError('Prepared adapter changed after preparation')
    if set(original) != set(actual):
        raise ValueError('Refinement changed the adapter file set')
    if any(original[k] != actual[k] for k in original if k != 'adapter_config.json'):
        raise ValueError('Refinement changed weights, tokenizer, or training state')
    config = json.loads((parent / 'adapter_config.json').read_text())
    expected = scaled_config(config, manifest['strength_multiplier'])
    if json.loads((candidate / 'adapter_config.json').read_text()) != expected:
        raise ValueError('Configuration change is not exactly the registered alpha scaling')
    state = json.loads((candidate / 'state.json').read_text())
    if state.get('step') != 32:
        raise ValueError('Expected the frozen step-32 DPO parent')
    return actual


def prepare(parent, destination, name, scale):
    parent, destination = Path(parent).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Refusing to replace an existing refinement')
    hashes = adapter_hashes(parent)
    config = scaled_config(json.loads((parent / 'adapter_config.json').read_text()), scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.refinement-') as temporary:
        stage = Path(temporary)
        shutil.copytree(parent, stage / 'adapter')
        (stage / 'adapter/adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
        manifest = {
            'schema_version': 1, 'candidate': name, 'parent_candidate': 'dpo-32',
            'parent_source': str(parent), 'strength_multiplier': scale,
            'prepared_at': datetime.now(timezone.utc).isoformat(),
            'parent_files_sha256': hashes, 'adapter_files_sha256': adapter_hashes(stage / 'adapter'),
            'scope': 'Prepared post-training refinement; no new optimizer updates or evaluation outcomes',
            'change': 'Only lora_alpha changes; weights, tokenizer, and step-32 state are preserved',
        }
        verify(parent, stage / 'adapter', manifest)
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(destination)
    return manifest
