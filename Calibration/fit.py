"""Fit a linear regression to values in summaries.txt.

Behavior:
- Read numeric values from `summaries.txt` (whitespace/tab-separated).
- Map y-values to x positions with spacing=1. By default we try to map indices
  start_x..end_x inclusive (1-based). For example start_x=2, end_x=400 maps
  to x = [2,3,...,400].
- If `summaries.txt` contains at least `end_x` values, the code will take the
  slice y[start_x-1:end_x] (1-based conversion). Otherwise, if it contains at
  least `N = end_x-start_x+1` values, it will take the first N values and map
  them to x=start_x..end_x.
- Fit linear regression (least-squares) using numpy.polyfit (degree=1).
- Print slope, intercept, R^2 and save a plot.

Usage: run this script in the same folder as `summaries.txt`.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


def read_numbers(path: Path):
    text = path.read_text(encoding='utf-8')
    # split on any whitespace (tabs/spaces/newlines)
    parts = text.strip().split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            # ignore non-numeric tokens
            continue
    return vals


def fit_linear(x, y):
    # Fit y = a*x + b
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    return a, b, r2, y_pred


def main(summaries_path='summaries.txt', start_x=2, end_x=400, out_png=None, cal_scale=1.0):
    p = Path(summaries_path)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    vals = read_numbers(p)
    if len(vals) == 0:
        print("No numeric values found in the file.")
        sys.exit(1)

    N_required = end_x - start_x + 1
    # Prefer slicing by 1-based indices if the file is long enough
    if len(vals) >= end_x:
        # use values corresponding to indices start_x..end_x (1-based)
        y = np.array(vals[start_x - 1:end_x])
        x = np.arange(start_x, end_x + 1)
        source = f'slice indices {start_x}..{end_x} from file (1-based)'
    elif len(vals) >= N_required:
        # file is shorter than end_x but has at least N_required values
        y = np.array(vals[:N_required])
        x = np.arange(start_x, end_x + 1)
        source = f'first {N_required} values from file mapped to x={start_x}..{end_x}'
    else:
        print(f"File contains only {len(vals)} numeric values, but need {N_required} for the requested range.")
        sys.exit(1)

    # We do NOT fit any linear model here. Instead use measured distances D from file
    print(f"Using: {source}")
    print(f"N points = {len(x)}")
    print("Note: skipping any linear fitting — plotting raw-derived values only.")
    # show a small table of raw D values
    n_show = min(10, len(x))
    print(f"\nDetailed raw distances (first {n_show} points):")
    print("x\t D (cm)")
    for xi, di in zip(x[:n_show], y[:n_show]):
        try:
            xi_print = int(xi)
        except Exception:
            xi_print = xi
        print(f"{xi_print}\t{di:.6f}")

    # Plot: do not plot any fitted lines — only show h vs t (formula and measured points)

    # --- Compute h from measured D and theoretical formula (no fitting anywhere)
    b_cm = 2.6  # transducer separation b in cm (from spec)
    # speed of sound at 25 C: v = 331 + 0.6*T => T=25 -> v = 346.13 m/s -> v_cm_per_us = 0.034613 cm/us
    v_cm_per_us = 0.034613

    D = y  # measured D values (cm)
    # measured h from geometry: h_measured = sqrt(D^2 - (b^2)/4)
    with np.errstate(invalid='ignore'):
        h_measured = np.sqrt(np.clip(D ** 2 - (b_cm ** 2) / 4.0, 0, None))

    # compute time-of-flight t (microseconds) from D using D = v * t / 2 -> t = 2*D / v
    t_us = 2.0 * D / v_cm_per_us

    # h from the provided numeric formula (***). Keep the formula line commented for checking
    # h = sqrt(1.1981e-3 * t^2 - 6.76/4)    <-- keep this line commented for checking
    with np.errstate(invalid='ignore'):
        h_from_formula = np.sqrt(np.clip(1.1981e-3 * (t_us ** 2) - 6.76 / 4.0, 0, None))

    # Plot h vs t: measured points and formula curve (no fits)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(t_us, h_from_formula, '-', color='purple', lw=2, label='h (formula ***)')
    ax2.scatter(t_us, h_measured, s=18, color='black', alpha=0.7, label='h (from D measured)')
    ax2.set_xlabel('Time-of-flight t (μs)')
    ax2.set_ylabel('h (cm)')
    ax2.set_title('Measured h from D and theoretical curve (***)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # save figure
    out_png2 = Path(__file__).with_name(f'height_vs_time_{start_x}_{end_x}.png')
    fig2.tight_layout()
    fig2.savefig(out_png2, dpi=150)
    print(f"Saved h vs t plot to: {out_png2}")

    if out_png is None:
        out_png = Path(__file__).with_name(f'linear_fit_{start_x}_{end_x}.png')
    else:
        out_png = Path(out_png)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear regression and MSE summary')
    parser.add_argument('--summaries', default='summaries.txt', help='path to summaries file')
    parser.add_argument('--start', type=int, default=2, help='start x (1-based)')
    parser.add_argument('--end', type=int, default=400, help='end x (1-based)')
    parser.add_argument('--out', default=None, help='output PNG path')
    parser.add_argument('--cal-scale', type=float, default=1.0, help='divide calibrated MSE by this factor for display')
    args = parser.parse_args()
    main(summaries_path=args.summaries, start_x=args.start, end_x=args.end, out_png=args.out, cal_scale=args.cal_scale)
