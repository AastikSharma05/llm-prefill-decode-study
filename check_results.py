"""Check the results for warning signs, then print the numbers your write-up needs.

Run it after each experiment (it skips whatever is missing):

  python check_results.py                                   # checks + key numbers
  python check_results.py --compare-a results_run1/exp_a.csv  # repeatability: earlier run vs this one

It also writes everything it prints to results/key_numbers.md, so the numbers you quote
can be traced back to a file in the repository.

Lines starting with "!!" are problems to fix (or understand) before trusting a run.
Lines starting with "??" are things to look at and be able to explain.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

OUT = []


def say(line=""):
    print(line)
    OUT.append(line)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def k(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- rough physics floors
# A back-of-envelope "fastest possible" time for Qwen2.5-7B, so you can see what fraction of
# the GPU's peak each measurement reached. If a measurement beats its floor (>100%), the
# measurement is wrong. These are estimates: check the arithmetic yourself before quoting it.
QWEN7B = {
    "non_embedding_params": 6.53e9,   # from the Qwen2.5 technical report
    "lm_head_params": 152064 * 3584,  # vocab x hidden; not tied to the embedding in the 7B
    "layers": 28,
    "attn_width": 28 * 128,           # query heads x head dim
    "kv_bytes_per_token": 2 * 28 * 4 * 128 * 2,
}
GPU_PEAKS = {"L4": (121e12, 300e9), "A100": (312e12, 1555e9)}  # dense bf16 FLOP/s, memory bytes/s


def prefill_flops(n):
    m = QWEN7B
    linear = 2 * m["non_embedding_params"] * n
    attention = 2 * m["layers"] * n * n * m["attn_width"]  # QK^T and AV, halved for the causal mask
    return linear + attention


def decode_bytes(context):
    m = QWEN7B
    weights = 2 * (m["non_embedding_params"] + m["lm_head_params"])  # every weight is read once per step
    return weights + m["kv_bytes_per_token"] * context


def gpu_peaks(results, args):
    if args.peak_tflops and args.mem_bw_gbs:
        return args.peak_tflops * 1e12, args.mem_bw_gbs * 1e9, "given on the command line"
    env = os.path.join(results, "env_a.txt")
    if os.path.exists(env):
        text = open(env).read()
        for name, peaks in GPU_PEAKS.items():
            if re.search(rf"\b{name}\b", text):
                return peaks[0], peaks[1], f"datasheet values for {name}"
    return None, None, None


# ---------------------------------------------------------------- GPU log
def check_gpu_log(path):
    if not os.path.exists(path):
        return
    clocks, temps = [], []
    for row in read_csv(path):
        vals = {key.strip(): v for key, v in row.items() if key}
        def num(prefix):
            for key, v in vals.items():
                if key.startswith(prefix):
                    m = re.search(r"[\d.]+", v or "")
                    return float(m.group()) if m else None
        util, clock, temp = num("utilization.gpu"), num("clocks.current.sm") or num("clocks.sm"), num("temperature.gpu")
        if util is not None and util >= 50 and clock:
            clocks.append(clock)
            if temp is not None:
                temps.append(temp)
    if not clocks:
        return say(f"?? {path}: no busy samples found")
    clocks.sort()
    p5 = clocks[int(0.05 * (len(clocks) - 1))]
    say(f"GPU while busy ({os.path.basename(path)}): SM clock median {med(clocks):.0f} MHz, "
        f"5th pct {p5:.0f} MHz, max temperature {max(temps) if temps else float('nan'):.0f} C")
    if p5 < 0.85 * med(clocks):
        say("?? the clock dropped by more than 15% at times. GPUs lower their clock when they reach their "
            "power limit, so some drop is normal; compare the two Exp A runs before trusting a trend.")


# ---------------------------------------------------------------- Experiment A
def exp_a(results, args):
    path = os.path.join(results, "exp_a.csv")
    if not os.path.exists(path):
        return say("skip Experiment A: no results/exp_a.csv"), None
    rows = read_csv(path)
    model = rows[0]["model"]
    say(f"## Experiment A  ({len(rows)} requests, model {model}, chunk budget {rows[0]['chunk_budget']})")
    g = defaultdict(list)
    for r in rows:
        g[(int(r["input_len"]), int(r["output_len"]))].append(r)
    ins = sorted({i for i, _ in g})
    outs = sorted({o for _, o in g})

    # Warning signs
    bad = 0
    for (i, o), rs in sorted(g.items()):
        t = [float(r["ttft_s"]) for r in rs]
        spread = (max(t) - min(t)) / med(t)
        if len(t) > 1 and spread > 0.10:
            bad += 1
            say(f"!! input {k(i)}, output {o}: TTFT repeats differ by {spread:.0%} (max-min over median). "
                "Something else may be using the GPU, or it is throttling.")
    if bad == 0:
        say("ok: every setting's TTFT repeats agree within 10%")
    ttft_in = {i: med([float(r["ttft_s"]) for (ii, _), rs in g.items() if ii == i for r in rs]) for i in ins}
    if len(ins) > 1:
        ratio = ttft_in[ins[-1]] / ttft_in[ins[0]]
        say(f"TTFT grows {ratio:.1f}x from {k(ins[0])} to {k(ins[-1])} input tokens "
            f"(input grew {ins[-1] / ins[0]:.0f}x)")
        if ratio < 0.25 * ins[-1] / ins[0]:
            say("!! TTFT barely grows with input length. The server is probably reusing cached work: "
                "check that run_exp_a.sh still passes --no-enable-prefix-caching.")
    if any(int(r["completion_tokens"]) != int(r["output_len"]) for r in rows):
        say("!! some requests returned the wrong number of tokens")
    multi = sum(int(r["multi_token_chunks"]) for r in rows)
    if multi:
        say(f"?? {multi} streamed chunks carried more than one token. With one request at a time this is "
            "unexpected; TPOT is still correct, but note it.")
    check_gpu_log(os.path.join(results, "gpu_log_a.csv"))

    # Key numbers
    say("")
    say("Numbers for Figure 1 (prefill vs decode):")
    m = {key: (med([float(r["ttft_s"]) for r in rs]), med([float(r["decode_s"]) for r in rs]),
               med([float(r["tpot_s"]) for r in rs])) for key, rs in g.items()}
    say("  median TTFT / decode time (ms):")
    say("  " + "input".rjust(7) + "".join(f"{str(o) + ' out':>20}" for o in outs))
    for i in ins:
        say("  " + k(i).rjust(7) + "".join(
            f"{m[(i, o)][0] * 1000:>9.0f} /{m[(i, o)][1] * 1000:>8.0f}" if (i, o) in m else f"{'-':>20}" for o in outs))
    big, small = ins[-1], outs[0]
    if (big, small) in m:
        t, d, _ = m[(big, small)]
        say(f"  - share of a {k(big)}-input, {small}-output request spent in prefill: {t / (t + d):.0%} "
            f"(TTFT {t * 1000:.0f} ms, decode {d * 1000:.0f} ms)")
    for o in outs:
        cross = next((i for i in ins if (i, o) in m and m[(i, o)][0] > m[(i, o)][1]), None)
        txt = f"from {k(cross)} input tokens" if cross else f"never, up to {k(ins[-1])} input tokens"
        say(f"  - prefill takes longer than decode for {o} output tokens: {txt}")
    tpot_in = {i: med([m[(i, o)][2] for o in outs if (i, o) in m]) for i in ins}
    say(f"  - TPOT (median over output lengths): {tpot_in[ins[0]] * 1000:.1f} ms at {k(ins[0])} input, "
        f"{tpot_in[ins[-1]] * 1000:.1f} ms at {k(ins[-1])} input "
        f"({tpot_in[ins[-1]] / tpot_in[ins[0]] - 1:+.0%}; each decode step reads a larger KV cache)")

    flops_peak, bw_peak, src = gpu_peaks(results, args)
    if "qwen2.5-7b" in model.lower() and flops_peak:
        say("")
        say(f"Rough share of the GPU's peak reached ({src}; YOUR check: redo this arithmetic by hand):")
        for i in ins:
            mfu = prefill_flops(i) / ttft_in[i] / flops_peak
            bw = decode_bytes(i) / tpot_in[i] / bw_peak
            say(f"  input {k(i):>4}: prefill {mfu:5.0%} of peak compute | decode {bw:5.0%} of peak memory bandwidth "
                f"(floor TPOT {decode_bytes(i) / bw_peak * 1000:.1f} ms)")
            if mfu > 1 or bw > 1:
                say("!! faster than the hardware allows: the measurement or the model of it is wrong")
    return say(""), model


def compare_a(old_path, results):
    new_path = os.path.join(results, "exp_a.csv")
    if not (os.path.exists(old_path) and os.path.exists(new_path)):
        return say(f"skip comparison: need both {old_path} and {new_path}")
    say(f"## Repeatability: {old_path} vs {new_path}")
    def medians(path):
        g = defaultdict(list)
        for r in read_csv(path):
            g[(int(r["input_len"]), int(r["output_len"]))].append(r)
        return {key: (med([float(r["ttft_s"]) for r in rs]), med([float(r["tpot_s"]) for r in rs])) for key, rs in g.items()}
    a, b = medians(old_path), medians(new_path)
    worst = 0
    for key in sorted(set(a) & set(b)):
        dt, dp = b[key][0] / a[key][0] - 1, b[key][1] / a[key][1] - 1
        worst = max(worst, abs(dt), abs(dp))
        flag = "!!" if max(abs(dt), abs(dp)) > 0.05 else "  "
        say(f"{flag} input {k(key[0]):>4}, output {key[1]:>3}: TTFT {dt:+.1%}, TPOT {dp:+.1%}")
    say(f"largest difference {worst:.1%}" + (" (over 5%: find out why before trusting either run)" if worst > 0.05 else
                                             " (within 5%)"))
    say("")


# ---------------------------------------------------------------- Experiment A2
def exp_a2(results):
    path = os.path.join(results, "exp_a2_transfer.csv")
    if not os.path.exists(path):
        return say("skip Experiment A2: no results/exp_a2_transfer.csv")
    rows = read_csv(path)
    say(f"## Experiment A2 (estimate, model {rows[0]['model_key']}, link efficiency {float(rows[0]['link_efficiency']):.0%})")
    by = {(int(r["input_len"]), int(r["bandwidth_gbps"])): r for r in rows}
    ins = sorted({i for i, _ in by})
    bws = sorted({b for _, b in by})
    target = 8192 if 8192 in ins else ins[len(ins) // 2]
    say("Numbers for Figure 2 (transfer vs prefill):")
    for bw in bws:
        r = by[(target, bw)]
        say(f"  - {k(target)} input at {bw} Gbit/s: adds {r['added_ttft_pct_layer_overlap']}% (ideal layer overlap, "
            f"{r['transfer_layer_overlap_ms']} ms) to {r['added_ttft_pct_no_overlap']}% (no overlap, "
            f"{r['transfer_no_overlap_ms']} ms) to the measured prefill of {r['prefill_ms']} ms")
    r = by[(target, bws[0])]
    kv_bytes = float(r["kv_mb"]) * 1e6
    eff = float(r["link_efficiency"])
    breakeven = kv_bytes * 8 / (float(r["prefill_ms"]) / 1000 * eff) / 1e9
    say(f"  - KV cache for {k(target)} input: {r['kv_mb']} MB; that is {kv_bytes / 64 / 1e6:.2f} MB shipped per "
        f"output token for 64 output tokens")
    say(f"  - with no overlap, transfer takes as long as the measured prefill on links slower than about "
        f"{breakeven:.0f} Gbit/s (nominal); this depends on this GPU's prefill speed")
    say("  Reminder: these are estimates from KV size and link speed, not measurements. Do not call either one a bound.")
    say("")


# ---------------------------------------------------------------- Experiment B
def exp_b(results):
    files = glob.glob(os.path.join(results, "exp_b", "*.json"))
    runs = []
    for f in files:
        r = json.load(open(f))
        m = re.match(r"^(chunk\d+(?:-lpt)?)_(off|colocated|kvremote)_r(\d+)$", r.get("label", ""))
        if m:
            r["_config"], r["_mode"], r["_rep"] = m.group(1), m.group(2), int(m.group(3))
            runs.append(r)
    if not runs:
        return say("skip Experiment B: no results/exp_b/<config>_<mode>_r<N>.json files")
    say(f"## Experiment B ({len(runs)} runs, model {runs[0]['model']})")
    cfgs = sorted({r["_config"] for r in runs}, key=lambda c: (int(re.search(r"\d+", c).group()), c))
    by = defaultdict(list)
    for r in runs:
        by[(r["_config"], r["_mode"])].append(r)

    # Warning signs
    for r in sorted(runs, key=lambda r: r["label"]):
        if r.get("possible_overload"):
            say(f"!! {r['label']}: summarization TTFT kept growing (overload). Rerun with a larger --sum-interval.")
        if "no chat-only run" in r.get("stall_threshold_source", ""):
            say(f"?? {r['label']}: stall threshold came from its own run, not the matching chat-only run")
        if r["_mode"] != "off" and r["summarization_completed"] < r["summarization_offered"]:
            say(f"?? {r['label']}: {r['summarization_completed']} of {r['summarization_offered']} summarization requests completed")
    for c in cfgs:
        co = [r["summarization_ttft"]["p50_ms"] for r in by[(c, "colocated")] if r["summarization_ttft"]["p50_ms"]]
        kv = [r["summarization_ttft"]["p50_ms"] for r in by[(c, "kvremote")] if r["summarization_ttft"]["p50_ms"]]
        if co and kv:
            ratio = med(kv) / med(co)
            line = (f"{c}: summarization TTFT median {med(co):.0f} ms colocated vs {med(kv):.0f} ms kvremote "
                    f"(ratio {ratio:.2f})")
            if ratio > 0.5:
                say("!! " + line + ". kvremote is not far below colocated, so the prefix cache is probably not "
                    "being reused and kvremote is NOT imitating disaggregation. Stop and investigate.")
            else:
                say("ok: " + line)
        offs = [r["chat_itl"]["p50_ms"] for r in by[(c, "off")]]
        if len(offs) > 1 and (max(offs) - min(offs)) / med(offs) > 0.10:
            say(f"?? {c}: chat-only median token gap varies {min(offs)}-{max(offs)} ms across repeats")

    # Key numbers
    say("")
    say("Numbers for Figure 3 (interference). mean [min-max] over repeats:")
    def cell(rs, getter, fmt="{:.0f}"):
        v = [getter(r) for r in rs if getter(r) is not None]
        if not v:
            return "-"
        return f"{fmt.format(statistics.mean(v))} [{fmt.format(min(v))}-{fmt.format(max(v))}]"
    metrics = [("p99 gap ms", lambda r: r["chat_itl"]["p99_ms"], "{:.0f}"),
               ("p99.9 gap ms", lambda r: r["chat_itl"]["p99_9_ms"], "{:.0f}"),
               ("stall s/user-min", lambda r: r["stall_seconds_per_user_minute"], "{:.2f}")]
    for c in cfgs:
        say(f"  {c}  (chat-only median gap {cell(by[(c, 'off')], lambda r: r['chat_itl']['p50_ms'], '{:.1f}')} ms)")
        for name, get, fmt in metrics:
            say(f"    {name:<17}" + "".join(f"{mode}: {cell(by[(c, mode)], get, fmt):<22}" for mode in ("off", "colocated", "kvremote")))
        for name, get, fmt in metrics:
            co = [get(r) for r in by[(c, "colocated")] if get(r) is not None]
            kv = [get(r) for r in by[(c, "kvremote")] if get(r) is not None]
            if co and kv:
                n = min(len(co), len(kv))
                overlap = min(co) <= max(kv) and min(kv) <= max(co)
                if n < 2:
                    verdict = "only one repeat each: no spread, so no claim yet"
                elif overlap:
                    verdict = "ranges OVERLAP: you cannot claim a difference"
                else:
                    verdict = (f"ranges do not overlap (with {n} repeats, say 'in all {n} repeats', "
                               "not 'significant')")
                say(f"    -> {name}: colocated minus kvremote = {fmt.format(statistics.mean(co) - statistics.mean(kv))}; {verdict}")
        counts = [r["chat_itl"]["count"] for r in by[(c, "colocated")]]
        if counts:
            say(f"    p99.9 of about {med(counts):.0f} chat gaps rests on its top ~{med(counts) / 1000:.0f} gaps per run")
    say("")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--compare-a", default=None, help="an earlier exp_a.csv to compare with results/exp_a.csv")
    p.add_argument("--peak-tflops", type=float, default=None, help="dense bf16 peak; default from env_a.txt (A100: 312)")
    p.add_argument("--mem-bw-gbs", type=float, default=None, help="memory bandwidth GB/s; default from env_a.txt (A100: 1555)")
    args = p.parse_args()

    mock = any("MOCK" in open(f).read() for f in
               glob.glob(os.path.join(args.results, "exp_a.csv")) + glob.glob(os.path.join(args.results, "exp_b", "*.json")))
    if mock:
        say("!! THESE ARE MOCK NUMBERS FROM mock_server.py. NEVER QUOTE THEM ANYWHERE.")
        say("")
    exp_a(args.results, args)
    if args.compare_a:
        compare_a(args.compare_a, args.results)
    exp_a2(args.results)
    exp_b(args.results)

    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, "key_numbers.md")
    with open(out, "w") as f:
        f.write("# Key numbers (written by check_results.py)\n\n```\n" + "\n".join(OUT) + "\n```\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
