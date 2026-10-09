import numpy as np
import matplotlib.pyplot as plt

# === 1. Đọc dữ liệu từ file .txt ===
with open('37cm_100.txt', 'r') as f:
    data = f.read().split()

values = np.array([float(x) for x in data[:100]])  # lấy 100 giá trị đầu

# === 2. Xác định min, max và bước nhảy ===
vmin, vmax = np.min(values), np.max(values)
step = (vmax - vmin) / 5
print(f"Min = {vmin}, Max = {vmax}, Step = {step}")

# === 3. Tạo biên cho 5 cột ===
bins = [vmin + i * step for i in range(6)]  # 6 biên → 5 khoảng
print("Biên cột:", bins)

# === 4. Vẽ histogram ===
plt.figure(figsize=(8, 5))
n, bins_plot, patches = plt.hist(values, bins=bins, color='dodgerblue', edgecolor='black', rwidth=0.9)

# === 5. Ghi số lượng trên đầu mỗi cột ===
for count, left, right in zip(n, bins_plot[:-1], bins_plot[1:]):
    if count > 0:
        plt.text((left + right) / 2, count + 0.5, int(count), ha='center', va='bottom', fontsize=10)

# === 6. Tùy chỉnh trục và nhãn ===
plt.title('Histogram chia 5 cột (không làm tròn giá trị)')
plt.xlabel('Khoảng giá trị (min → max)')
plt.ylabel('Số lượng (Frequency)')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.xticks([(bins[i] + bins[i+1]) / 2 for i in range(5)],
           [f"[{bins[i]:.5f}, {bins[i+1]:.5f})" for i in range(5)],
           rotation=30, ha='right')
plt.tight_layout()
plt.show()
