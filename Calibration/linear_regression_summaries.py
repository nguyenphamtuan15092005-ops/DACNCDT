"""
Fit a linear regression to values in summaries.txt.

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

Phù hợp hồi quy tuyến tính với các giá trị trong summaries.txt.

Hành vi:
- Đọc các giá trị số từ `summaries.txt` (được phân cách bằng khoảng trắng/tab).
- Ánh xạ các giá trị y đến các vị trí x với khoảng cách = 1. Theo mặc định, chúng tôi cố gắng ánh xạ các chỉ số
start_x..end_x bao gồm (dựa trên 1). Ví dụ: start_x=2, end_x=400 ánh xạ
đến x = [2,3,...,400].
- Nếu `summaries.txt` chứa ít nhất các giá trị `end_x`, mã sẽ lấy lát cắt
y[start_x-1:end_x] (chuyển đổi dựa trên 1). Ngược lại, nếu nó chứa ít nhất
các giá trị `N = end_x-start_x+1`, mã sẽ lấy N giá trị đầu tiên và ánh xạ
chúng đến x=start_x..end_x.
- Phù hợp hồi quy tuyến tính (bình phương nhỏ nhất) bằng cách sử dụng numpy.polyfit (bậc = 1).
- In độ dốc, giao điểm, R^2 và lưu biểu đồ.

Cách sử dụng: chạy tập lệnh này trong cùng thư mục với `summaries.txt`
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

    a, b, r2, y_pred = fit_linear(x, y)

    print(f"Using: {source}")
    print(f"N points = {len(x)}")
    print(f"Slope (a) = {a:.6f}")
    print(f"Intercept (b) = {b:.6f}")
    print(f"R^2 = {r2:.6f}")

    # Print the linear equation in a clear form and a small detailed table
    print(f"Equation: y = {a:.6f}*x + {b:.6f}")
    n_show = min(10, len(x))
    print(f"\nDetailed values (first {n_show} points):")
    print("x\t y\t y_pred")
    for xi, yi, ypi in zip(x[:n_show], y[:n_show], y_pred[:n_show]):
        # xi may be numpy type - ensure readable integer when appropriate
        try:
            xi_print = int(xi)
        except Exception:
            xi_print = xi
        print(f"{xi_print}\t{yi:.6f}\t{ypi:.6f}")

    # --- Drift fitting and calibration (C(x) = F(x) - D(x))
    # Residual between fitted output and true value
    residual_fit = y_pred - x
    # Fit a linear drift D(x) to residual_fit vs x
    d_slope, d_intercept = np.polyfit(x, residual_fit, 1)
    D_x = d_slope * x + d_intercept
    # Calibrated function C(x) = F(x) - D(x)
    C_x = y_pred - D_x

    # MSE of calibrated function C(x) compared to true x
    mse_C_of_x = float(np.mean((C_x - x) ** 2))

    # Also try applying drift subtraction to observed y: y_obs_cal = y - D(x)
    y_obs_cal = y - D_x
    mse_obs_cal = float(np.mean((y_obs_cal - x) ** 2))

    # ALSO fit drift using observed residuals (y - x) — often more appropriate
    residual_obs = y - x
    d_obs_slope, d_obs_intercept = np.polyfit(x, residual_obs, 1)
    D_obs_x = d_obs_slope * x + d_obs_intercept
    y_obs_cal2 = y - D_obs_x
    mse_obs_cal2 = float(np.mean((y_obs_cal2 - x) ** 2))

    # Print equations for clarity
    print('\nDrift fit from observed residuals D_obs(x):')
    print(f'D_obs(x) = {d_obs_slope:.6f}*x + {d_obs_intercept:.6f}')
    print(f'Applying y_corrected = y - D_obs(x) gives MSE = {mse_obs_cal2:.6f}')

    # Print equations for clarity
    print('\nDrift fit D(x):')
    print(f'D(x) = {d_slope:.6f}*x + {d_intercept:.6f}')
    print('Calibrated function C(x) = F(x) - D(x):')
    # C(x) will be linear with coefficients (a - d_slope, b - d_intercept)
    print(f'C(x) = {(a - d_slope):.6f}*x + {(b - d_intercept):.6f}')

    print('\nCalibration MSEs:')
    print(f'| MSE of C(x) vs x           | {mse_C_of_x:9.4f} |')
    print(f'| MSE of (y - D(x)) vs x     | {mse_obs_cal:9.4f} |')

    # Compute Mean Squared Errors
    # Actual MSE: between observed output y and true value x
    mse_actual = float(np.mean((y - x) ** 2))

    # Calibrated MSE (inverse calibration): apply inverse of fitted model to observed outputs
    # i.e. y_calibrated = (y - b) / a  -> compare to true x
    if a == 0:
        mse_calibrated = float('nan')
        calibrated_note = ' (cannot compute, slope a == 0)'
        y_calibrated = None
    else:
        y_calibrated = (y - b) / a
        mse_calibrated = float(np.mean((y_calibrated - x) ** 2))
        calibrated_note = ''

    # Alternative: MSE of the fitted prediction (y_pred = a*x + b) compared to true x
    mse_predicted = float(np.mean((y_pred - x) ** 2))

    # Optionally scale calibrated MSE (user requested dividing by factor to "correct")
    try:
        cal_scale_f = float(cal_scale)
    except Exception:
        cal_scale_f = 1.0
    if cal_scale_f == 0:
        calibrated_scaled = float('nan')
        scale_note = ' (scale=0 invalid)'
    else:
        calibrated_scaled = mse_calibrated / cal_scale_f if not np.isnan(mse_calibrated) else float('nan')
        scale_note = '' if cal_scale_f == 1.0 else f' (divided by {cal_scale_f})'

    # Print MSEs in a compact table
    print('\nMSE results:')
    print('+---------------------------+-----------+')
    print(f'| Actual MSE                | {mse_actual:9.4f} |')
    print(f'| Calibrated MSE (inverse)  | {mse_calibrated:9.4f} |{calibrated_note}')
    print(f'| Fitted-prediction MSE     | {mse_predicted:9.4f} |')
    if cal_scale_f != 1.0:
        print(f'| Calibrated MSE scaled     | {calibrated_scaled:9.4f} |{scale_note}')
    print('+---------------------------+-----------+')

    # Plot: only plot the fitted line, ideal line, and calibrated line (lines only)
    fig, ax = plt.subplots(figsize=(11, 6))
    # fitted linear model
    ax.plot(x, y_pred, color='red', lw=2, label='Linear Fit')
    # ideal line y=x
    y_ideal = x
    ax.plot(x, y_ideal, color='green', lw=2, ls='--', label='Ideal Fit')
    # calibrated observed (using observed-residual drift D_obs)
    y_calibrated_obs = y - D_obs_x
    ax.plot(x, y_calibrated_obs, color='tab:blue', lw=1.8, ls=':', label='Calibrated')
    # calibrated function C(x) = F(x) - D(x)
    C_x = y_pred - D_x
    ax.plot(x, C_x, color='navy', lw=1.6, ls='-.', label='C(x) = F(x)-D(x)')

    # (Equation box removed — plot shows only lines and legend)

    ax.set_xlabel('True Value Data (cm)')
    ax.set_ylabel('Output Value Data calibrated(cm)')
    ax.set_title(f'Linear regression ({start_x}..{end_x}), N={len(x)}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

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
