"""Compute and plot drift between Output and True values using `summaries.txt`.

- Maps y-values from `summaries.txt` to x = 2..400 (same mapping used earlier).
- Computes drift = y - x (output - true).
- Saves drift values to `summaries_drift_2_400.txt` (one per line).
- Plots drift vs x as points, overlays an OLS fitted line for the drift, and a horizontal
  zero line for reference. Saves PNG `drift_2_400.png`.
- Prints basic statistics (mean drift, std drift) and fit parameters.

Usage: run in the same directory as `summaries.txt`.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys


def read_numbers(path: Path):
    text = path.read_text(encoding='utf-8')
    parts = text.strip().split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            continue
    return vals


def fit_linear(x, y):
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    return a, b, r2, y_pred


def main(summaries_path='summaries.txt', start_x=2, end_x=400):
    p = Path(summaries_path)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    vals = read_numbers(p)
    if len(vals) == 0:
        print("No numeric values found in the file.")
        sys.exit(1)

    N_required = end_x - start_x + 1
    if len(vals) >= end_x:
        y = np.array(vals[start_x - 1:end_x])
        x = np.arange(start_x, end_x + 1)
        source = f'slice indices {start_x}..{end_x} from file (1-based)'
    elif len(vals) >= N_required:
        y = np.array(vals[:N_required])
        x = np.arange(start_x, end_x + 1)
        source = f'first {N_required} values from file mapped to x={start_x}..{end_x}'
    else:
        print(f"File contains only {len(vals)} numeric values, but need {N_required} for the requested range.")
        sys.exit(1)

    # compute drift (output - true)
    drift = y - x

    # save drift
    out_drift = Path(__file__).with_name(f'summaries_drift_{start_x}_{end_x}.txt')
    np.savetxt(out_drift, drift, fmt='%.8f')
    print(f"Saved drift values to: {out_drift} (N={len(drift)})")

    # stats
    mean_d = float(np.mean(drift))
    std_d = float(np.std(drift, ddof=1)) if len(drift) > 1 else float('nan')
    print(f"Mean drift = {mean_d:.6f}")
    print(f"Std (sample) drift = {std_d:.6f}")

    # fit linear model to drift vs x
    a, b, r2, drift_pred = fit_linear(x, drift)
    print(f"Drift fit slope = {a:.8f}, intercept = {b:.8f}, R^2 = {r2:.6f}")

    # plot
    plt.figure(figsize=(9, 5))
    plt.scatter(x, drift, s=8, color='tab:blue', alpha=0.7, label='drift points')
    plt.plot(x, drift_pred, color='tab:red', lw=2, label=f'fit: drift={a:.4e}x+{b:.4f}')
    plt.axhline(0.0, color='gray', lw=1, ls='--', label='zero reference')
    plt.xlabel('True Value (x)')
    plt.ylabel('Drift (output - true)')
    plt.title(f'Drift vs True Value ({start_x}..{end_x})')
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_png = Path(__file__).with_name(f'drift_{start_x}_{end_x}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


if __name__ == '__main__':
    main()
