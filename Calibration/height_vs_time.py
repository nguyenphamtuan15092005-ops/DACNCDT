"""
Compute height h from measured distances in summaries.txt and plot h vs time.

This standalone script keeps things simple (no fitting):
- Reads numeric values from summaries.txt (horizontal or whitespace separated).
- Interprets values as distances D in cm.
- Computes time-of-flight t (microseconds) using D = v * t / 2 with v determined by temperature.
- Computes measured h: h_measured = sqrt(D^2 - (b^2)/4)
- Computes theoretical h from provided formula (commented in the code):
    #      <-- retained as a comment for verification
- Saves a plot and optionally a CSV of the computed columns.

Usage:
    python height_vs_time.py --summaries summaries.txt --start 2 --end 400 --out height.png --csv out.csv

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv


def read_numbers(path: Path):
    text = path.read_text(encoding='utf-8')
    parts = text.strip().split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            continue
    return np.array(vals)


def main(summaries='summaries.txt', start=2, end=400, out='height_vs_time.png', outcsv=None, b=2.6, temp=25.0):
    p = Path(summaries)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    vals = read_numbers(p)
    N_required = end - start + 1
    if len(vals) >= end:
        D = vals[start - 1:end]
        x = np.arange(start, end + 1)
        source = f'slice indices {start}..{end} from file (1-based)'
    elif len(vals) >= N_required:
        D = vals[:N_required]
        x = np.arange(start, end + 1)
        source = f'first {N_required} values from file mapped to x={start}..{end}'
    else:
        raise SystemExit(f"File contains only {len(vals)} numeric values, but need {N_required} for the requested range.")

    print(f"Using: {source}")
    print(f"N points = {len(x)}")

    # speed of sound v (m/s) = 331 + 0.6 * T (°C)
    v_m_per_s = 331.0 + 0.6 * float(temp)
    # convert to cm per microsecond: m/s -> cm/us => (m/s) * 100 / 1e6 = 1e-4 * (m/s)
    # so v_cm_per_us = v_m_per_s * 1e-4
    v_cm_per_us = v_m_per_s * 1e-4

    # measured D are in cm (array D)
    D = np.array(D, dtype=float)

    # measured h from geometry
    with np.errstate(invalid='ignore'):
        h_measured = np.sqrt(np.clip(D ** 2 - (float(b) ** 2) / 4.0, 0, None))

    # time-of-flight in microseconds
    t_us = 2.0 * D / v_cm_per_us

    # theoretical formula (kept commented for verification in code):
    # h = sqrt(1.1981e-3 * t^2 - 6.76/4)    <-- keep this line commented for checking
    with np.errstate(invalid='ignore'):
        h_formula = np.sqrt(np.clip(1.1981e-3 * (t_us ** 2) - 6.76 / 4.0, 0, None))

    # Plot: top = h vs t (formula + measured h), bottom = time vs distance points
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={'height_ratios': [2, 1]})

    # Top: h vs t
    ax1.plot(t_us, h_formula, '-', color='purple', lw=2, label='h (formula)')
    ax1.scatter(t_us, h_measured, s=30, color='black', alpha=0.8, label='h (from D measured)')
    ax1.set_xlabel('Time-of-flight t (μs)')
    ax1.set_ylabel('h (cm)')
    ax1.set_title('Measured h from D and theoretical curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom: time vs distance points (t vs D) and ideal D(t) line
    ax2.scatter(t_us, D, s=18, color='tab:blue', alpha=0.8, label='Measured D (cm)')
    # ideal relation D = v * t / 2  -> plot line
    t_line = np.linspace(np.nanmin(t_us), np.nanmax(t_us), 200)
    D_line = v_cm_per_us * t_line / 2.0
    ax2.plot(t_line, D_line, color='green', lw=1.8, ls='--', label='Ideal D = v*t/2')
    ax2.set_xlabel('Time-of-flight t (μs)')
    ax2.set_ylabel('D (cm)')
    ax2.set_title('Time-of-flight vs Measured Distance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    outp = Path(out)
    fig.savefig(outp, dpi=150)
    print(f"Saved plot to: {outp}")

    # Optionally save CSV
    if outcsv:
        outc = Path(outcsv)
        header = ['index', 'D_cm', 't_us', 'h_measured_cm', 'h_formula_cm']
        with outc.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for idx, (di, ti, hi_m, hi_f) in enumerate(zip(D, t_us, h_measured, h_formula), start=start):
                writer.writerow([idx, f"{di:.6f}", f"{ti:.6f}", f"{hi_m:.6f}", f"{hi_f:.6f}"])
        print(f"Saved CSV to: {outc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot h vs time from measured D values')
    parser.add_argument('--summaries', default='summaries.txt')
    parser.add_argument('--start', type=int, default=2)
    parser.add_argument('--end', type=int, default=400)
    parser.add_argument('--out', default='height_vs_time.png')
    parser.add_argument('--csv', default=None, help='optional CSV output path')
    parser.add_argument('--b', type=float, default=2.6, help='transducer separation b in cm')
    parser.add_argument('--temp', type=float, default=25.0, help='temperature in Celsius for speed of sound')
    args = parser.parse_args()
    main(summaries=args.summaries, start=args.start, end=args.end, out=args.out, outcsv=args.csv, b=args.b, temp=args.temp)
