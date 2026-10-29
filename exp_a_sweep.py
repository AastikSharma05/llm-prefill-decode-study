"""Experiment A: how long are prefill and decode for different input and output lengths?

Sends ONE request at a time (nothing else running), so the timings show the pure cost
of each phase without interference from other users. The (input, output, repeat) runs
are shuffled, so slow drift on the GPU (heat, clock changes) does not line up with
input length and fake a trend.

Example:
  python exp_a_sweep.py --base-url http://localhost:8000 --chunk-budget 2048 --out results/exp_a.csv
"""
import argparse
import asyncio
import csv
import os
import random
import statistics
from collections import defaultdict

import aiohttp

from common import get_model_name, random_prompt, stream_request


async def main(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model or await get_model_name(session, args.base_url)
        print(f"model: {model}")

        # Warm-up: the first few requests are slow (CUDA graphs, memory allocation).
        print(f"warm-up: {args.warmup} requests (not recorded)")
        for i in range(args.warmup):
            await stream_request(session, args.base_url, model, random_prompt(2048, seed=10_000 + i), 32)

        plan = [(i, o, r) for i in args.input_lens for o in args.output_lens for r in range(args.repeats)]
        random.Random(args.seed).shuffle(plan)
        rows = []
        for n, (in_len, out_len, rep) in enumerate(plan, 1):
            r = await stream_request(session, args.base_url, model, random_prompt(in_len, seed=n), out_len)
            if not r["length_ok"]:
                raise SystemExit(f"asked for {out_len} tokens, got {r['completion_tokens']}: is ignore_eos working?")
            rows.append({
                "model": model,
                "chunk_budget": args.chunk_budget,
                "order": n,
                "input_len": in_len,
                "output_len": out_len,
                "repeat": rep,
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "ttft_s": round(r["ttft_s"], 5),
                "decode_s": round(r["decode_s"], 5),
                "tpot_s": round(r["tpot_s"], 6),
                "e2e_s": round(r["e2e_s"], 5),
                "multi_token_chunks": r["multi_token_chunks"],
            })
            if n % 10 == 0 or n == len(plan):
                print(f"  {n}/{len(plan)} requests done")

    groups = defaultdict(list)
    for x in rows:
        groups[(x["input_len"], x["output_len"])].append(x)
    for (in_len, out_len), g in sorted(groups.items()):
        ttft = [x["ttft_s"] for x in g]
        print(f"input {in_len:>6} | output {out_len:>4} | "
              f"TTFT median {statistics.median(ttft) * 1000:8.1f} ms "
              f"(min {min(ttft) * 1000:.1f}, max {max(ttft) * 1000:.1f}) | "
              f"decode median {statistics.median(x['decode_s'] for x in g) * 1000:8.1f} ms | "
              f"TPOT median {statistics.median(x['tpot_s'] for x in g) * 1000:6.2f} ms")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default=None, help="model name; read from the server if not given")
    p.add_argument("--input-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--output-lens", type=int, nargs="+", default=[32, 64, 128, 512])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="seed for the shuffled run order")
    p.add_argument("--chunk-budget", type=int, default=None,
                   help="the server's --max-num-batched-tokens, recorded in the CSV")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", default="results/exp_a.csv")
    asyncio.run(main(p.parse_args()))
