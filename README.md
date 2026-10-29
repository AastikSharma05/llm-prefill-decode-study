# Prefill vs decode for long-input, short-output LLM requests

> **Rewrite this README in your own words once you have real results.** The sections below are a skeleton.

## Question

Prefill/decode disaggregation (DistServe, Splitwise, Mooncake) moves a request's KV cache from a
prefill GPU to a decode GPU. For summarization-style requests (long input, short output) that cache
is large while the decode phase is short. This repository measures both sides of that trade-off on a
single GPU:

- **Experiment A:** prefill time (TTFT) and decode time for input lengths 1k to 16k tokens and output lengths 32 to 512 tokens.
- **Experiment A2:** KV cache size and the time to ship it at 10 to 400 Gbit/s, as extra latency on top of the measured prefill (with and without layer-wise overlap).
- **Experiment B:** chat users' token gaps on one GPU under three workloads: chat only; chat plus summarization done fully on this GPU (colocated); chat plus summarization whose KV cache is already present, so only its decode runs here (emulated disaggregation, via prefix caching). Repeated for three prefill chunk budgets, three times each.

## Setup

- GPU: _(fill in, e.g. 1x NVIDIA L4 24 GB, JarvisLabs)_
- Model: _(e.g. Qwen/Qwen2.5-7B-Instruct, bf16)_
- vLLM version: _(output of `pip show vllm`)_
- Date of runs: _(fill in)_

## Results

_(Add figures from `figures/` and two or three sentences per figure: what it shows, with the numbers.)_

## Limitations

- One GPU and one model size. Transfer time is estimated from KV size and link speed, not measured; real systems land between the no-overlap and layer-overlap estimates, and per-message overheads can make them slower.
- Faster GPUs shrink prefill and decode times but not network time, so on datacenter GPUs the transfer would be a larger share than on this one.
- "kvremote" emulates disaggregation with prefix caching on one GPU; it reproduces the decode load but not the network transfer or the memory used by arriving caches.
- Prompts are random token IDs. Prefill cost depends on length, not content, but output text is meaningless.

## Reproduce

Step-by-step for a rented GPU, including which vLLM version to install: [RUN_ON_GPU.md](RUN_ON_GPU.md).

```bash
pip install -r requirements.txt     # plus `pip install "vllm==<version>"` on the GPU machine
bash run_exp_a.sh                   # about 40 minutes on one 24 GB GPU
bash run_exp_b.sh                   # about 75 minutes (3 chunk budgets x 3 workloads x 3 repeats)
CHUNKS=512 LPT=1 bash run_exp_b.sh  # optional, about 25 minutes
python plot_results.py              # figures/
python check_results.py             # warning signs + results/key_numbers.md
```

Practice without a GPU: `MOCK=1 bash run_exp_a.sh --repeats 2` uses `mock_server.py`, which produces fake numbers stamped "MOCK".

## AI assistance

_(Keep a line like this, honestly worded.)_ The benchmark scripts were written with help from an AI assistant. I ran all experiments, checked the code, and wrote the analysis.
