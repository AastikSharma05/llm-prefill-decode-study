# Running the experiments on a rented GPU

Copy-paste steps for one NVIDIA L4 (24 GB). GPU time if nothing breaks: about 20 min setup,
5 min smoke test, 40 min Experiment A, 75 min Experiment B (+25 min optional). Budget 4 hours.

Lines starting with `#` inside code blocks are comments; you do not type them.

## 0. Before you rent (on your laptop)

Delete the mock output, then push the code, so the GPU machine starts clean:

```bash
rm -rf results figures logs
git add . && git commit -m "Add check_results.py and GPU runbook" && git push
```

## 1. Rent the machine

L4, 24 GB, PyTorch template, at least 50 GB of disk. Open a terminal on it.

## 2. Check the machine first (decides which vLLM to install)

```bash
nvidia-smi          # write down: GPU name, "Driver Version", and "CUDA Version" (top right)
python3 --version
df -h               # find the disk with ~50 GB free
```

Pick the vLLM version from the **CUDA Version** that `nvidia-smi` shows:

| `nvidia-smi` shows | Install | Why |
|---|---|---|
| CUDA Version 13.0 or higher | `vllm==0.31.0` | vLLM 0.20 and newer install a PyTorch built for CUDA 13, which needs a driver that supports CUDA 13 |
| CUDA Version 12.x | `vllm==0.18.0` | the newest vLLM whose PyTorch (2.10) is built for CUDA 12 |

Why not the newest, 0.32.0? It was released on 18 October 2026 and needs Python 3.11 or newer.
0.31.0 works with Python 3.10. Both 0.31.0 and 0.18.0 have every flag these scripts use
(checked in their released source: `--long-prefill-token-threshold`, `--no-enable-prefix-caching`,
`--generation-config`, and the `return_token_ids` request field). **Use the same version for every run, including the rerun in step 7.**

## 3. Install

```bash
git clone https://github.com/<your-username>/llm-prefill-decode-study.git
cd llm-prefill-decode-study
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install "vllm==0.31.0" -r requirements.txt          # or 0.18.0, from the table above
python -c "import torch; print(torch.__version__, 'CUDA', torch.version.cuda, 'GPU ok:', torch.cuda.is_available())"
```

The last line must end in `GPU ok: True`. If it prints `False`, or an error says the NVIDIA
driver is too old, you picked the wrong row: `pip install "vllm==0.18.0"` and check again.

**Where the model is stored.** It goes to `~/.cache/huggingface` (about 15 GB). If `df -h ~`
shows less than 30 GB free there, put it on the big disk instead (replace the path):

```bash
echo 'export HF_HOME=/path/on/big/disk/hf' >> ~/.bashrc && source ~/.bashrc
```

Download the model once on its own, so a slow download does not look like a hung server:

```bash
hf download Qwen/Qwen2.5-7B-Instruct      # older installs: huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

**Start tmux**, so a closed browser tab does not kill a run. The new tmux shell does not have
your venv, so activate it again inside:

```bash
tmux new -s exp            # if "command not found": apt-get update && apt-get install -y tmux
cd ~/llm-prefill-decode-study && source venv/bin/activate     # adjust the path if you cloned elsewhere
```

Detach with Ctrl+B then D. Come back with `tmux attach -t exp`.

## 4. Smoke test (about 8 minutes): catch problems before the long runs

```bash
bash run_exp_a.sh --repeats 1 --input-lens 1024 16384 --output-lens 32 --warmup 1
CHUNKS=2048 REPEATS=1 bash run_exp_b.sh --warmup 5 --duration 30
python check_results.py
```

What you want to see:

- `results/env_a.txt` names the L4, your vLLM version and `torch ... (CUDA ...)`.
- TTFT at 16k is many times TTFT at 1k.
- TPOT is above the "floor TPOT" line that `check_results.py` prints (about 47 to 50 ms on an L4).
  Work out why that floor exists before you continue: every decode step reads all ~14 GB of
  weights, and the L4 reads memory at 300 GB/s.
- An `ok:` line for kvremote vs colocated summarization TTFT. A `!!` line there means the
  "KV arrives" mode is not imitating disaggregation; stop and send me the output.

Then clear the smoke-test output so it cannot mix with the real runs:

```bash
rm -rf results figures
```

## 5. The real runs

```bash
bash run_exp_a.sh            # about 40 minutes
python check_results.py      # read every !! line before going on
bash run_exp_b.sh            # about 75 minutes
python check_results.py
CHUNKS=512 LPT=1 bash run_exp_b.sh    # optional, about 25 minutes
```

- If a repeat spread of more than 10% is flagged in Experiment A, run `nvidia-smi` and check that
  only your vLLM process is on the GPU, then rerun Experiment A.
- If Experiment B prints `WARNING: ... overloaded`, note which chunk budget it was, then rerun
  just that one with more space between summarization requests, for example
  `CHUNKS=512 bash run_exp_b.sh --sum-interval 10`. Say in the README that you did this, and why.
- The GPU bills while you wait. You can detach from tmux, but do not stop the instance yet.

## 6. Save everything, then stop the instance

```bash
python plot_results.py
python check_results.py                         # writes results/key_numbers.md
git add results figures
git commit -m "L4 results, vLLM $(pip show vllm | grep Version | cut -d' ' -f2)"
git push                                        # password = a GitHub personal access token
tar czf results_backup.tgz results figures      # also download this file from the file browser
```

Check that the push appeared on GitHub, then **stop the instance**. Check whether your provider
still bills for the disk of a stopped instance; if it does and you will not reuse it, delete it.

## 7. Repeatability rerun (a few days later, about 1 hour including setup)

Same GPU type and the same vLLM version. Run Experiment A's full sweep again, compare, and keep
run 1 as the main result (Experiment B was spaced using run 1):

```bash
mkdir -p results/run1_a results/run2_a
cp results/exp_a.csv results/exp_a2_transfer.csv results/gpu_log_a.csv results/env_a.txt results/run1_a/
bash run_exp_a.sh
cp results/exp_a.csv results/exp_a2_transfer.csv results/gpu_log_a.csv results/env_a.txt results/run2_a/
python check_results.py --compare-a results/run1_a/exp_a.csv | tee results/repeatability.txt
cp results/run1_a/* results/                    # put run 1 back as the main result
python check_results.py                         # key_numbers.md for run 1 again
git add results && git commit -m "Repeatability rerun of Experiment A" && git push
```

Differences under 5% mean the setup is repeatable. Above 5%, compare the two `gpu_log_a.csv`
files (clock and temperature) before you trust either run.

## 8. What to send me

`results/key_numbers.md`, the three PNGs in `figures/`, and any `!!` lines you could not explain.
If a server fails to start, the last 30 lines of its log from `logs/`, exactly as printed.
