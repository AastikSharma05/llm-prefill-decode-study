"""Turn the result files into three figures for the README and the proposal.

  python plot_results.py --results results --out figures

Writes:
  figures/fig1_prefill_vs_decode.png   Experiment A
  figures/fig2_transfer_vs_prefill.png Experiment A2
  figures/fig3_interference.png        Experiment B
Any figure whose input file is missing is skipped.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Colour-blind-checked categorical order (blue, orange, aqua, yellow).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, QUIET, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": QUIET, "axes.labelcolor": INK, "xtick.color": QUIET,
    "ytick.color": QUIET, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})


def k(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


def is_mock(model):
    return "MOCK" in str(model)


def title(ax, text, model):
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold")
    if is_mock(model):
        watermark(ax.figure)


def watermark(fig):
    """Stamp mock results so they can never be mistaken for real measurements."""
    fig.text(0.5, 0.5, "MOCK DATA - NOT REAL", ha="center", va="center", fontsize=34,
             color="#e34948", alpha=0.18, rotation=20, fontweight="bold")


def fig1(results, out):
    path = os.path.join(results, "exp_a.csv")
    if not os.path.exists(path):
        return print("skip fig1: no exp_a.csv")
    df = pd.read_csv(path)
    model = df["model"].iloc[0]
    med = df.groupby(["output_len", "input_len"])[["ttft_s", "decode_s"]].median().reset_index()
    outs = sorted(med["output_len"].unique())
    fig, axes = plt.subplots(1, len(outs), figsize=(4.2 * len(outs), 3.6), sharey=True)
    axes = axes if len(outs) > 1 else [axes]
    for ax, o in zip(axes, outs):
        d = med[med["output_len"] == o]
        x = range(len(d))
        ax.bar(x, d["ttft_s"], color=C[0], width=0.6, label="prefill (TTFT)", edgecolor="white", linewidth=1)
        ax.bar(x, d["decode_s"], bottom=d["ttft_s"], color=C[1], width=0.6, label="decode",
               edgecolor="white", linewidth=1)
        ax.set_xticks(list(x), [k(i) for i in d["input_len"]])
        ax.set_xlabel("input length (tokens)")
        ax.set_title(f"{o} output tokens", color=INK, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median time per request (s)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Prefill vs decode time, one request at a time ({model})",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    if is_mock(model):
        watermark(fig)
    fig.savefig(os.path.join(out, "fig1_prefill_vs_decode.png"))
    print("wrote fig1")


def fig2(results, out):
    path = os.path.join(results, "exp_a2_transfer.csv")
    if not os.path.exists(path):
        return print("skip fig2: no exp_a2_transfer.csv")
    df = pd.read_csv(path)
    a_path = os.path.join(results, "exp_a.csv")
    model = pd.read_csv(a_path)["model"].iloc[0] if os.path.exists(a_path) else df["model_key"].iloc[0]
    eff = df["link_efficiency"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pre = df.drop_duplicates("input_len").sort_values("input_len")
    ax.plot(pre["input_len"], pre["prefill_ms"], color=INK, linewidth=2.5, marker="s", markersize=5)
    labels = [(pre["prefill_ms"].iloc[-1], "measured prefill (TTFT)", "bold")]
    for i, (bw, g) in enumerate(sorted(df.groupby("bandwidth_gbps"), key=lambda t: t[0])):
        g = g.sort_values("input_len")
        ax.plot(g["input_len"], g["transfer_no_overlap_ms"], color=C[i % 4], linewidth=2, marker="o", markersize=4)
        labels.append((g["transfer_no_overlap_ms"].iloc[-1], f"KV transfer, {bw} Gbit/s", "normal"))
    # Direct labels at the line ends, nudged apart (in log space) so they never overlap.
    import math
    labels.sort()
    placed = []
    for y, text, weight in labels:
        ly = math.log10(y)
        if placed and ly - placed[-1] < 0.12:
            ly = placed[-1] + 0.12
        placed.append(ly)
        ax.annotate(text, (pre["input_len"].iloc[-1], y), xytext=(pre["input_len"].iloc[-1] * 1.12, 10 ** ly),
                    textcoords="data", va="center", color=INK, fontsize=8.5, fontweight=weight)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = sorted(df["input_len"].unique())
    ax.set_xticks(ticks, [k(t) for t in ticks])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("input length (tokens)")
    ax.set_ylabel("milliseconds")
    ax.set_xlim(right=max(ticks) * 3.2)
    title(ax, f"KV transfer time (no overlap, {eff:.0%} link efficiency) vs measured prefill\n({model})", model)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_transfer_vs_prefill.png"))
    print("wrote fig2")


MODES = [("off", "chat only"), ("colocated", "+ summarization, prefill here (colocated)"),
         ("kvremote", "+ summarization, KV arrives (disaggregated)")]


def fig3(results, out):
    rows = []
    for f in glob.glob(os.path.join(results, "exp_b", "*.json")):
        r = json.load(open(f))
        m = re.match(r"^(chunk\d+(?:-lpt)?)_(off|colocated|kvremote)_r(\d+)$", r["label"])
        if m:
            rows.append({"config": m.group(1), "mode": m.group(2), "rep": int(m.group(3)),
                         "p99": r["chat_itl"]["p99_ms"], "p999": r["chat_itl"]["p99_9_ms"],
                         "stall": r["stall_seconds_per_user_minute"], "model": r["model"]})
    if not rows:
        return print("skip fig3: no results/exp_b/<config>_<mode>_r<N>.json files")
    df = pd.DataFrame(rows)
    model = df["model"].iloc[0]
    configs = sorted(df["config"].unique(), key=lambda c: (int(re.search(r"\d+", c).group()), c))
    panels = [("p99", "chat token gap, 99th pct (ms)"), ("p999", "chat token gap, 99.9th pct (ms)"),
              ("stall", "stall time per chat user (s/min)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    w = 0.27
    for ax, (col, name) in zip(axes, panels):
        for j, (mode, label) in enumerate(MODES):
            xs, mean, lo, hi = [], [], [], []
            for i, c in enumerate(configs):
                v = df[(df["config"] == c) & (df["mode"] == mode)][col]
                if v.empty:
                    continue
                xs.append(i + (j - 1) * w)
                mean.append(v.mean())
                lo.append(v.mean() - v.min())
                hi.append(v.max() - v.mean())
            ax.bar(xs, mean, width=w - 0.03, color=C[j], label=label,
                   yerr=[lo, hi], error_kw={"ecolor": QUIET, "elinewidth": 1, "capsize": 2})
        ax.set_xticks(range(len(configs)), [c.replace("chunk", "") for c in configs])
        ax.set_xlabel("chunk budget, --max-num-batched-tokens")
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"What summarization costs chat users on one GPU ({model}); bars = mean, whiskers = min-max of repeats",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    if is_mock(model):
        watermark(fig)
    fig.savefig(os.path.join(out, "fig3_interference.png"))
    print("wrote fig3")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.results, a.out)
    fig2(a.results, a.out)
    fig3(a.results, a.out)
