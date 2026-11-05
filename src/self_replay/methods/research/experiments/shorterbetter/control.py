"""Synchronized training control; Telegram credentials never enter a GPU worker."""
import json
from pathlib import Path
import shutil
import time

import torch
from transformers import TrainerCallback
from .distributed import atomic_json
from .monitor import summarize
from .plot import render
from .stopping import decide, apply_quality_policy


class ScaledControl(TrainerCallback):
    def __init__(self, output, protocol, lease, accelerator):
        self.output, self.protocol, self.lease = Path(output), protocol, lease
        self.accelerator = accelerator
        self.best = None
        self.started = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # Propagate SIGTERM/budget requests from any rank before the next
        # generation collective, and retain all-rank optimizer/RNG checkpoints.
        reserve = self.protocol.get('execution', {}).get('shutdown_reserve_seconds', 600)
        stop = control.should_training_stop or time.time() >= (self.lease.get('hard_deadline') or float('inf')) - reserve
        request = self.output.parent / 'stop-request.json'
        if request.exists():
            value = json.loads(request.read_text())
            stop |= value.get('job_name') == self.lease['job_name']
        flag = torch.tensor(int(stop), device=self.accelerator.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        if flag.item():
            control.should_training_stop = control.should_save = True
            if (self.protocol.get('execution', {}).get('terminal_evaluation_required')
                    and args.eval_strategy != 'no' and state.global_step > 0):
                control.should_evaluate = True
            if state.is_world_process_zero:
                path = self.output / 'decision.json'
                prior = json.loads(path.read_text()) if path.exists() else {}
                if not prior.get('stop'):
                    completed = state.global_step >= args.max_steps
                    atomic_json(path, {'stop': True, 'reason': 'registered_updates_completed' if completed else 'deadline_or_stop_request',
                                      'optimizer_step': state.global_step})
        if state.is_world_process_zero:
            atomic_json(self.output / 'progress.json', {'optimizer_step': state.global_step,
                'rollout_batches': state.global_step / 4, 'updated_at': time.time(),
                'elapsed_seconds': time.time() - self.started})
            if (self.protocol['execution'].get('paired_short_experiment')
                    and not self.protocol['execution'].get('multidomain_full_verification')
                    and state.global_step % 4 == 0):
                from .live_metrics import publish_series
                publish_series(self.output, state.global_step)
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self.accelerator.wait_for_everyone()
        payload = [None]
        if state.is_world_process_zero:
            paths = [self.output / f'rank-{r:03d}' / 'rollouts.jsonl'
                     for r in range(self.accelerator.num_processes)]
            report = summarize(paths, expected_questions=self.protocol['data']['monitor_subset_questions'])
            if report['checkpoints'][-1]['optimizer_step'] != state.global_step:
                raise RuntimeError('Monitoring did not include current checkpoint')
            decision = decide(report, self.protocol['stopping'])
            if self.protocol['execution'].get('multidomain'):
                from .multidomain.prepare import DOMAINS
                domain_reports = {d: summarize(paths, expected_questions=self.protocol['data']['monitor_subset_questions']//4, task=d) for d in DOMAINS}
                domain_decisions = {d: decide(r, self.protocol['stopping']) for d, r in domain_reports.items()}
                report['domains'] = domain_reports
                decision['domain_decisions'] = domain_decisions
                regressions = [d for d, v in domain_decisions.items() if v['stop'] and v['reason']=='persistent_regression']
                if regressions:
                    decision.update(stop=True, reason='persistent_regression', regressed_domains=regressions)
                if any(v['best_eligible_step'] != state.global_step for v in domain_decisions.values()):
                    decision['best_eligible_step'] = None
            elif self.protocol['execution']['phase'] == 'benchmark':
                decision.update(stop=False, reason='throughput_benchmark_not_efficacy')
            decision = apply_quality_policy(decision, self.protocol)
            prior_path = self.output / 'decision.json'
            if self.protocol['execution'].get('terminal_evaluation_required') and prior_path.exists():
                prior = json.loads(prior_path.read_text())
                if prior.get('stop') and prior.get('reason') == 'deadline_or_stop_request':
                    decision.update(stop=True, reason=prior['reason'], quality_reason=decision['reason'])
            atomic_json(self.output / 'learning-curve.json', report)
            render(report, self.output / 'learning-curve.png')
            for domain, domain_report in report.get('domains', {}).items():
                render(domain_report, self.output / ('learning-curve-'+domain+'.png'))
            atomic_json(self.output / 'decision.json', decision)
            from .throughput import forecast
            # Tiny CPU callback fixtures deliberately have no billing lease.
            if 'created_at' in self.lease:
                atomic_json(self.output / 'throughput.json', forecast(self.output, self.protocol, self.lease))
            event = {'event': 'validation', 'time': time.time(), 'metrics': report['checkpoints'][-1],
                     'decision': decision}
            with (self.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            payload[0] = decision
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        self.best = payload[0]['best_eligible_step']
        if payload[0]['stop']:
            control.should_training_stop = control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.accelerator.wait_for_everyone()
        if (state.is_world_process_zero and self.protocol['execution'].get('paired_short_experiment')
                and not self.protocol['execution'].get('multidomain_full_verification')
                and state.global_step % (4*self.protocol['execution']['preserve_every_rollouts']) == 0):
            from .live_metrics import preserve_snapshot
            preserve_snapshot(self.output, state.global_step)
        if state.is_world_process_zero:
            with (self.output / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps({'event': 'checkpoint_saved', 'time': time.time(),
                                         'optimizer_step': state.global_step}) + '\n')
        if (state.is_world_process_zero and self.best == state.global_step
                and not self.protocol['execution'].get('paired_short_experiment')
                and not self.protocol['execution'].get('multidomain_full_verification')):
            source = self.output / f'checkpoint-{state.global_step}'
            target = self.output / 'best-checkpoint'
            temporary = self.output / 'best-checkpoint.pending'
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(source, temporary)
            if target.exists():
                shutil.rmtree(target)
            temporary.rename(target)
            atomic_json(self.output / 'best-checkpoint.json', {'optimizer_step': self.best,
                'scope': 'Development-selected, needs independent confirmation'})
        self.accelerator.wait_for_everyone()
