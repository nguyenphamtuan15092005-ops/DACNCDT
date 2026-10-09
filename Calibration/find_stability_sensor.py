import matplotlib.pyplot as plt
import numpy as np


with open('23cm.txt', 'r') as f:
    data = f.read().split()

values = np.array([float(x) for x in data])

mean_value = np.mean(values)

true_value = 23.0


plt.figure(figsize=(10, 6))
x = np.arange(1, len(values) + 1)
plt.scatter(
    x, values,
    s=2,                    # kích thước chấm
    color='blue',
    alpha=0.8,
    marker='x',             
    label='Data Points'
)

plt.axhline(y=mean_value, color='orange', linestyle='--', label=f'Mean = {mean_value:.2f}')

plt.title('Biểu đồ 1000 giá trị (23cm) - Scatter Plot')
plt.xlabel('Lần đo (Index)')
plt.ylabel('Giá trị (Value)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

plt.show()
