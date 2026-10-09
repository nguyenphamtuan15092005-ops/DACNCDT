import matplotlib.pyplot as plt
import numpy as np

# === 1. Đọc dữ liệu từ file .txt ===
with open('23cm.txt', 'r') as f:
    data = f.read().split()

# Chuyển thành mảng số (float)
values = np.array([float(x) for x in data])

# === 2. Tính giá trị trung bình ===
mean_value = np.mean(values)

# === 3. Giá trị thật (true_value) ===
true_value = 23.0

# === 4. Vẽ biểu đồ dạng chấm với marker tùy chọn ===
plt.figure(figsize=(10, 6))
x = np.arange(1, len(values) + 1)
plt.scatter(
    x, values,
    s=2,                    # kích thước chấm
    color='blue',
    alpha=0.8,
    marker='x',              # ← 's' = hình vuông (thay đổi ở đây)
    label='Data Points'
)

# === 5. Thêm hai đường ngang ===
plt.axhline(y=mean_value, color='orange', linestyle='--', label=f'Mean = {mean_value:.2f}')
# plt.axhline(y=true_value, color='red', linestyle='--', label=f'True Value = {true_value:.2f}')

# === 6. Tùy chỉnh biểu đồ ===
plt.title('Biểu đồ 1000 giá trị (23cm) - Scatter Plot')
plt.xlabel('Lần đo (Index)')
plt.ylabel('Giá trị (Value)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

# === 7. Hiển thị ===
plt.show()
