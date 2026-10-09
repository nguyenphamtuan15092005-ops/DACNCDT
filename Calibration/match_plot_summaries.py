"""Create a plot that matches the attached style: only the fitted linear fit (red) and the ideal y=x (dashed green).

- Reads numeric values from `summaries.txt` (whitespace separated).
- Maps the first 399 values to x = 2..400.
- Fits an OLS line y = a*x + b and plots the fitted line (red) and y=x (green dashed).
- Saves PNG `match_linear_ideal_2_400.png` in the same folder.
- Prints slope/intercept/R^2 for reference.

Run: python match_plot_summaries.py
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
    elif len(vals) >= N_required:
        y = np.array(vals[:N_required])
        x = np.arange(start_x, end_x + 1)
    else:
        print(f"File contains only {len(vals)} numeric values, but need {N_required} for the requested range.")
        sys.exit(1)

    a, b, r2, y_pred = fit_linear(x, y)

    print(f"N points = {len(x)}")
    print(f"Slope (a) = {a:.6f}")
    print(f"Intercept (b) = {b:.6f}")
    print(f"R^2 = {r2:.6f}")

    # Create the plot (line only) to match attached style
    fig, ax = plt.subplots(figsize=(8, 5))

    # Fitted line (red)
    ax.plot(x, y_pred, color='red', lw=2, label='Linear Fit')

    # Ideal line y = x (dashed green)
    ax.plot(x, x, color='green', lw=2, ls='--', label='Ideal Fit')

    # Labels and grid to match the sample
    ax.set_xlabel('True Value Data (cm)')
    ax.set_ylabel('Output Value Data (cm)')
    ax.set_title('')
    ax.grid(True, alpha=0.3)

    # Make axes start at 0 and extend a bit beyond max for similar framing
    max_x = max(x.max(), y_pred.max())
    upper = max_x + 20
    ax.set_xlim(0, end_x + 20)
    ax.set_ylim(0, upper)

    ax.legend(loc='upper left')
    plt.tight_layout()

    out_png = Path(__file__).with_name(f'match_linear_ideal_{start_x}_{end_x}.png')
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


if __name__ == '__main__':
    main()
