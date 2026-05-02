"""
Plot stats from live_solve_stats.jsonl files (Easy vs Extreme).

Requires: pip install matplotlib

Example:
  .venv/bin/python -m scripts.plot_live_stats \\
    --easy 'data/EASY - live_solve_stats.jsonl' \\
    --extreme data/live_solve_stats.jsonl \\
    --out-dir data/plots
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# JSON field remains `infer_ms` (client HTTP round-trip until `/predict` returns).
API_SOLUTION_TIME = "Time to solution from API"
API_SOLUTION_TIME_AXIS = f"{API_SOLUTION_TIME} (s)"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_mode(rows: list[dict]) -> tuple[int, float, float, float]:
    """Returns (n_games, win_rate_pct, mean_api_solution_s, mean_total_s)."""
    if not rows:
        return 0, float("nan"), float("nan"), float("nan")
    n = len(rows)
    wins = sum(1 for r in rows if r.get("ok"))
    win_rate = 100.0 * wins / n
    mean_infer_ms = sum(float(r.get("infer_ms", 0)) for r in rows) / n
    mean_total_ms = sum(float(r.get("total_ms", 0)) for r in rows) / n
    return n, win_rate, mean_infer_ms / 1000.0, mean_total_ms / 1000.0


def print_summary(easy: list[dict], extreme: list[dict]) -> None:
    print()
    print("--- Summary ---")
    for label, rows in ("Easy", easy), ("Extreme", extreme):
        n, wr, api_s, total_s = summarize_mode(rows)
        if n == 0:
            print(f"{label}: no games in log")
            continue
        print(f"{label} (n={n}):")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  Mean time to solution from API: {api_s:.2f} s")
        print(f"  Mean total time per game (capture + API + fill): {total_s:.2f} s")
    print()


def cumulative_success(ok_flags: list[bool]) -> list[float]:
    out: list[float] = []
    acc = 0
    for i, ok in enumerate(ok_flags, start=1):
        acc += 1 if ok else 0
        out.append(100.0 * acc / i)
    return out


def plot_all(easy: list[dict], extreme: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def series(label: str, rows: list[dict]) -> tuple[list[int], list[bool], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = list(range(1, len(rows) + 1))
        ok = [bool(r.get("ok")) for r in rows]
        api_ms = np.array([float(r.get("infer_ms", 0)) for r in rows])
        total = np.array([float(r.get("total_ms", 0)) for r in rows])
        clues = np.array([float(r.get("clues", 0)) for r in rows])
        api_ms_success = api_ms[np.array(ok)]
        return idx, ok, api_ms, total, clues, api_ms_success

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rows, color in (
        ("Easy", easy, "tab:green"),
        ("Extreme", extreme, "tab:red"),
    ):
        if not rows:
            continue
        idx, ok_flags, _, _, _, _ = series(name, rows)
        cum = cumulative_success(ok_flags)
        ax.plot(idx, cum, label=f"{name} cumulative success %", color=color)
    ax.set_xlabel("Game index")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success rate over games")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "success_rates.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name, rows in (
        (axes[0], "Easy", easy),
        (axes[1], "Extreme", extreme),
    ):
        if not rows:
            ax.set_visible(False)
            continue
        _, _, api_ms, _, _, api_ok = series(name, rows)
        ax.hist(api_ms / 1000.0, bins=25, alpha=0.55, color="steelblue", label="All games")
        if len(api_ok):
            ax.hist(api_ok / 1000.0, bins=25, alpha=0.55, color="darkgreen", label="Successful only")
        ax.set_xlabel(API_SOLUTION_TIME_AXIS)
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: {API_SOLUTION_TIME}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "api_solution_time_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows, marker in (
        ("Easy", easy, "o"),
        ("Extreme", extreme, "x"),
    ):
        if len(rows) < 2:
            continue
        _, ok_flags, api_ms, _, clues, _ = series(name, rows)
        ok_arr = np.array(ok_flags)
        ax.scatter(
            clues[ok_arr],
            api_ms[ok_arr] / 1000.0,
            alpha=0.6,
            marker=marker,
            label=f"{name} success",
        )
        ax.scatter(
            clues[~ok_arr],
            api_ms[~ok_arr] / 1000.0,
            alpha=0.35,
            marker=marker,
            facecolors="none",
            edgecolors="gray",
            label=f"{name} failure",
        )
    ax.set_xlabel("Detected clues")
    ax.set_ylabel(API_SOLUTION_TIME_AXIS)
    ax.set_title(f"Detected clues vs {API_SOLUTION_TIME}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clues_vs_api_solution_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    reasons = {
        "Easy": Counter(str(r.get("reason", "unknown")) for r in easy),
        "Extreme": Counter(str(r.get("reason", "unknown")) for r in extreme),
    }
    all_reasons = sorted(set().union(*(r.keys() for r in reasons.values())))
    x = np.arange(len(all_reasons))
    w = 0.35
    for i, (label, rows) in enumerate((("Easy", easy), ("Extreme", extreme))):
        if not rows:
            continue
        counts = [reasons[label].get(rr, 0) for rr in all_reasons]
        ax.bar(x + (i - 0.5) * w, counts, width=w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(all_reasons, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Outcome reasons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reasons.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Easy vs Extreme live solve JSONL stats.")
    p.add_argument("--easy", type=Path, required=True, help="JSONL for Easy mode.")
    p.add_argument("--extreme", type=Path, required=True, help="JSONL for Extreme mode.")
    p.add_argument("--out-dir", type=Path, default=Path("data/plots"), help="Output PNG folder.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    easy_rows = load_jsonl(args.easy)
    extreme_rows = load_jsonl(args.extreme)
    plot_all(easy_rows, extreme_rows, args.out_dir)
    print(f"Saved plots under {args.out_dir.resolve()}")
    print_summary(easy_rows, extreme_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
