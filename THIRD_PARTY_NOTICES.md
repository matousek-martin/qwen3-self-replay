# Third-party attribution

The project's Apache-2.0 license applies to original code and released adapters; it does not replace upstream or dataset terms.

| Component | Source | Recorded license / handling |
|---|---|---|
| Qwen3-0.6B | [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0; pinned base revision in model cards |
| ThinkingCap evaluator | [BottleCapAI/ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test) | Apache-2.0; history and license retained in vendor/thinkingcap |
| ShorterBetter reward reference | [ShorterBetter paper](https://arxiv.org/abs/2504.21370) | Apache-2.0 header retained; source commit and file hash in vendor/shorterbetter/manifest.json |
| vLLM compatibility fixture | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0; pinned 0.11.0 source fixture and hash checks retained |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Recorded MIT |
| APPS | [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps) | Recorded MIT; source problem provenance retained |
| Tulu instruction-following | [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | Recorded ODC-BY; attribution retained |
| Glaive function calling | [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Recorded Apache-2.0 |
| xLAM function calling | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Recorded CC-BY-4.0; source revision retained |
| SVAMP | [arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP) | MIT; evaluation only |
| MBPP / MBPP+ | [Google MBPP](https://github.com/google-research/google-research/tree/master/mbpp), [EvalPlus](https://github.com/evalplus/mbppplus_release) | CC-BY-4.0 / Apache-2.0; evaluation only |
| BFCL | [Gorilla](https://github.com/ShishirPatil/gorilla) | Apache-2.0; evaluation only |
| IFEval | [Google Research](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Apache-2.0; evaluation only |

Training data is normalized, screened and split as documented in provenance/training-data.json and provenance/data-selection.json. These are modified selections, not an unchanged upstream dataset release. Frozen prepared data is a separately attributed research asset, never covered by a blanket software license.

Historical methods also refer to Skywork, CommonsenseQA, ARC, OpenCoder, KodCode and TACO. Their exact source records remain with the relevant preparation modules/configurations. The Skywork aggregate license was unknown in the original record: do not assume it inherits a code repository's license or redistribute it as Apache-2.0.
