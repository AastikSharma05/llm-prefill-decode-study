"""Experiment B: how much do summarization requests slow down chat users on the same GPU?

Several "chat users" keep sending short-input, long-output requests. We record every gap
between their output tokens (inter-token latency, ITL). Three workloads are compared:

  off        chat users only. The baseline.
  colocated  chat users + summarization requests (long input, short output) that do their
             whole prefill AND decode on this GPU. This is "no disaggregation".
  kvremote   chat users + summarization requests whose KV cache is already on this GPU,
             so only their decode runs here. This imitates "disaggregation": prefill happened
             on another GPU and the cache arrived over the network. It works by sending the
             SAME long prompt every time with prefix caching on, so the server reuses the
             cached KV instead of recomputing it. (Needs --enable-prefix-caching on the server.)

  colocated - kvremote = the interference that disaggregation would remove.
  kvremote  - off      = the decode load that stays on this GPU either way.

Example:
  python exp_b_interference.py --mode colocated --label chunk2048_colocated_r1
"""
import argparse
import asyncio
import csv
import itertools
import json
import os
import random
import statistics
import time
import zlib

import aiohttp

from common import get_model_name, percentile, random_prompt, stream_request

_seed = itertools.count(1_000_000)
FIXED_SUMMARY_SEED = 424242  # the one prompt reused in kvremote mode


def auto_interval(args, model):
    """Space summarization arrivals so their prefill uses about a third of the GPU's time.

    Too many arrivals overload the GPU and the queue grows for the whole run, which
    measures overload rather than interference.
    """
    try:
        with open(args.exp_a) as f:
            rows = list(csv.DictReader(f))
        if rows[0]["model"] != model:
            raise SystemExit(f"{args.exp_a} was measured on '{rows[0]['model']}', but the server runs "
                             f"'{model}'. Rerun Experiment A first, or pass --sum-interval.")
        lens = sorted({int(r["input_len"]) for r in rows}, key=lambda n: abs(n - args.sum_input))
        ttft = statistics.median(float(r["ttft_s"]) for r in rows if int(r["input_len"]) == lens[0])
        return max(2.0, round(3 * ttft, 2)), f"3 x TTFT({lens[0]} tokens) from {args.exp_a}"
    except (FileNotFoundError, KeyError, ValueError, IndexError):
        return 3.0, "default (no Experiment A results found)"


async def chat_user(session, args, model, start_delay, window, gaps, chat_ttfts, multi, rng):
    await asyncio.sleep(start_delay)  # stagger users so they do not move in lockstep
    record_after, stop_at = window
    while time.perf_counter() < stop_at:
        r = await stream_request(session, args.base_url, model,
                                 random_prompt(args.chat_input, next(_seed)), args.chat_output)
        # Keep only gaps that END inside the recording window.
        gaps.extend(g for t, g in r["token_gaps"] if record_after <= t <= stop_at)
        if record_after <= r["t_send"] <= stop_at:
            chat_ttfts.append(r["ttft_s"])
            multi[0] += r["multi_token_chunks"]
        await asyncio.sleep(rng.uniform(0, args.think_time))


async def summarizer(session, args, model, window, results):
    record_after, stop_at = window
    tasks = []

    offered = 0

    async def one(t_arrival):
        seed = FIXED_SUMMARY_SEED if args.mode == "kvremote" else next(_seed)
        r = await stream_request(session, args.base_url, model, random_prompt(args.sum_input, seed), args.sum_output)
        if t_arrival >= record_after:
            results.append({"t_rel_s": round(t_arrival - record_after, 2), "ttft_s": r["ttft_s"], "e2e_s": r["e2e_s"]})

    # Fixed spacing between arrivals, so every run sees the same offered load.
    while time.perf_counter() < stop_at - args.sum_interval:
        await asyncio.sleep(args.sum_interval)
        now = time.perf_counter()
        offered += now >= record_after
        tasks.append(asyncio.create_task(one(now)))
    await asyncio.gather(*tasks)
    return offered


def stats(values_s):
    def ms(x):
        return None if x is None else round(x * 1000, 2)
    return {
        "p50_ms": ms(percentile(values_s, 50)),
        "p90_ms": ms(percentile(values_s, 90)),
        "p99_ms": ms(percentile(values_s, 99)),
        "p99_9_ms": ms(percentile(values_s, 99.9)),
        "max_ms": ms(max(values_s)) if values_s else None,
        "count": len(values_s),
    }


def stall_threshold(args, own_p50):
    """Stalls = token gaps longer than twice the median gap of the chat-only run with the same
    server setting and repeat, so the threshold is identical across the three workloads."""
    if args.stall_threshold_ms is not None:
        return args.stall_threshold_ms / 1000, "set by --stall-threshold-ms"
    if args.mode == "off":
        return 2 * (own_p50 or 0.0), "2 x this run's median gap"
    off_label = args.label.replace(f"_{args.mode}_", "_off_")
    try:
        with open(os.path.join(args.out_dir, f"{off_label}.json")) as f:
            return 2 * json.load(f)["chat_itl"]["p50_ms"] / 1000, f"2 x median gap of {off_label}"
    except (FileNotFoundError, KeyError, TypeError):
        return 2 * (own_p50 or 0.0), "2 x this run's median gap (no chat-only run found)"


async def main(args):
    global _seed
    # Every run gets its own prompts. With prefix caching on, reusing prompts from an earlier
    # run would let the server skip work and make later runs look faster.
    _seed = itertools.count((zlib.crc32(args.label.encode()) % 100_000) * 1_000_000)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        model = args.model or await get_model_name(session, args.base_url)
        if args.sum_interval is None:
            args.sum_interval, why = auto_interval(args, model)
        else:
            why = "set by --sum-interval"
        if args.mode == "kvremote":
            # Put the fixed prompt's KV cache on the GPU once, before anything is recorded.
            await stream_request(session, args.base_url, model,
                                 random_prompt(args.sum_input, FIXED_SUMMARY_SEED), 1)
        start = time.perf_counter()
        window = (start + args.warmup, start + args.warmup + args.duration)
        print(f"model: {model} | mode {args.mode} | {args.chat_users} chat users | "
              f"summarization every {args.sum_interval} s ({why}) | "
              f"{args.warmup:.0f} s warm-up + {args.duration:.0f} s recorded")

        gaps, chat_ttft, sums, multi = [], [], [], [0]
        jobs = [chat_user(session, args, model, rng.uniform(0, 2.0), window, gaps, chat_ttft, multi,
                          random.Random(args.seed * 100 + i))
                for i in range(args.chat_users)]
        if args.mode != "off":
            jobs.append(summarizer(session, args, model, window, sums))
        done = await asyncio.gather(*jobs)
    offered = done[-1] if args.mode != "off" else 0

    threshold, threshold_why = stall_threshold(args, percentile(gaps, 50))
    stalls = [g for g in gaps if g > threshold]
    user_minutes = args.chat_users * args.duration / 60
    half = args.duration / 2
    early = [s["ttft_s"] for s in sums if s["t_rel_s"] < half]
    late = [s["ttft_s"] for s in sums if s["t_rel_s"] >= half]
    overload = bool(early and late and statistics.median(late) > 1.5 * statistics.median(early))

    result = {
        "label": args.label,
        "mode": args.mode,
        "model": model,
        "chat_users": args.chat_users,
        "chat_input": args.chat_input,
        "chat_output": args.chat_output,
        "sum_input": args.sum_input,
        "sum_output": args.sum_output,
        "sum_interval_s": args.sum_interval,
        "duration_s": args.duration,
        "chat_itl": stats(gaps),
        "chat_ttft": stats(chat_ttft),
        "stall_threshold_ms": round(threshold * 1000, 2),
        "stall_threshold_source": threshold_why,
        "stalls_per_user_minute": round(len(stalls) / user_minutes, 2),
        "stall_seconds_per_user_minute": round(sum(stalls) / user_minutes, 3),
        "summarization_ttft": stats([s["ttft_s"] for s in sums]),
        "summarization_offered": offered,
        "summarization_completed": len(sums),
        "possible_overload": overload,
        "chat_multi_token_chunks": multi[0],
    }
    with open(os.path.join(args.out_dir, f"{args.label}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.label}_itl.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["itl_ms"])
        w.writerows([[round(x * 1000, 3)] for x in gaps])
    if sums:
        with open(os.path.join(args.out_dir, f"{args.label}_summaries.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t_rel_s", "ttft_s", "e2e_s"])
            w.writeheader()
            w.writerows(sums)

    c = result["chat_itl"]
    print(f"chat ITL  p50 {c['p50_ms']} ms | p99 {c['p99_ms']} ms | p99.9 {c['p99_9_ms']} ms | "
          f"stalls/user-min {result['stalls_per_user_minute']} | samples {c['count']}")
    if sums:
        print(f"summarization: {len(sums)} completed of {offered} offered | TTFT median "
              f"{statistics.median(s['ttft_s'] for s in sums) * 1000:.0f} ms")
    if overload:
        print("WARNING: summarization TTFT kept growing during the run. The GPU is probably overloaded; "
              "raise --sum-interval and rerun.")
    print(f"saved {args.out_dir}/{args.label}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["off", "colocated", "kvremote"], required=True)
    p.add_argument("--label", required=True, help="name for this run, e.g. chunk2048_colocated_r1")
    p.add_argument("--chat-users", type=int, default=8)
    p.add_argument("--chat-input", type=int, default=256)
    p.add_argument("--chat-output", type=int, default=256)
    p.add_argument("--think-time", type=float, default=1.0, help="max random pause between a user's requests (s)")
    p.add_argument("--sum-input", type=int, default=8192)
    p.add_argument("--sum-output", type=int, default=64)
    p.add_argument("--sum-interval", type=float, default=None,
                   help="seconds between summarization requests; default: 3 x TTFT from Experiment A")
    p.add_argument("--exp-a", default="results/exp_a.csv")
    p.add_argument("--warmup", type=float, default=15)
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stall-threshold-ms", type=float, default=None,
                   help="default: 2 x median gap of the matching chat-only run")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out-dir", default="results/exp_b")
    asyncio.run(main(p.parse_args()))
