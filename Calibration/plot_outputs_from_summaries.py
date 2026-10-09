"""Plot output values from `summaries.txt` mapped to x = 2..400 as points and overlay an OLS fit line.

Saves `outputs_points_2_400.png` and prints slope/intercept/R^2.
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

    a, b, r2, y_pred = fit_linear(x, y)

    print(f"Using: {source}")
    print(f"N points = {len(x)}")
    print(f"Slope (a) = {a:.6f}")
    print(f"Intercept (b) = {b:.6f}")
    print(f"R^2 = {r2:.6f}")

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=12, color='purple', marker='x', label='data points')
    plt.plot(x, y_pred, color='red', lw=2, label=f'fit: y={a:.4f}x+{b:.4f}')
    plt.xlabel('True Value (x)')
    plt.ylabel('Output Value')
    plt.title(f'Outputs mapped to x={start_x}..{end_x} (N={len(x)})')
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_png = Path(__file__).with_name(f'outputs_points_{start_x}_{end_x}.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


if __name__ == '__main__':
    main()
