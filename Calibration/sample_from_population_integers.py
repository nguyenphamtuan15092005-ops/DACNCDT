
import numpy as np
from math import sqrt
from pathlib import Path
# Draw gausse
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
# -------------------------
# Tham số cấu hình (có thể thay đổi)
# -------------------------
# True mean và sigma của phân phối (giả sử chiều cao phân phối chuẩn)
true_mean = 170.0
sigma = 7.0
# Kích thước quần thể (số cá thể trong population)
N_pop = 10000
# Kích thước mẫu mỗi lần rút
n = 4

# Số lần Monte Carlo (số mẫu độc lập sẽ rút để ước lượng trung bình các chỉ số)
MC = 10000

# Seeds để làm cho kết quả có thể tái tạo
# seed_population: seed dùng để sinh quần thể
# seed_sampling: seed dùng để rút mẫu nhiều lần
seed_population = 42
seed_sampling = 123

# -------------------------
# Tạo quần thể (integer-valued population)
# -------------------------
# Tạo bộ sinh ngẫu nhiên có seed để kết quả reproducible
rng = np.random.default_rng(seed_population)

# Sinh N_pop giá trị từ N(true_mean, sigma)
pop = rng.normal(loc=true_mean, scale=sigma, size=N_pop)

# Làm tròn các giá trị về số nguyên (theo yêu cầu của bạn lấy phần tử nguyên)
pop_int = np.rint(pop).astype(int)  # round to nearest integer

# print(f'Population size: {N_pop}, integer values from ~N({true_mean},{sigma})')
# print('Population mean (rounded):', pop_int.mean(), 'population std (rounded):', pop_int.std(ddof=1))

# -------------------------
# Chuẩn bị rút mẫu (sampling)
# -------------------------
# Tạo RNG riêng cho rút mẫu, dùng seed_sampling để kết quả rút mẫu lặp lại
rng2 = np.random.default_rng(seed_sampling)

# Mảng để lưu giá trị SD mỗi lần lặp cho 3 ước lượng
s1_sds = np.empty(MC)  # SD từ variance chia (n-1)
s2_sds = np.empty(MC)  # SD từ variance chia n
s3_sds = np.empty(MC)  # SD tính quanh true mean và chia n

# -------------------------
# Vòng lặp Monte Carlo: rút mẫu và tính 3 ước lượng
# -------------------------
for i in range(MC):
    # rút n phần tử từ population, WITHOUT replacement (replace=False)
    sample = rng2.choice(pop_int, size=n, replace=False)

    # trung bình mẫu
    xbar = sample.mean()

    # tổng bình phương sai so với trung bình mẫu: sum((xi - xbar)^2)
    ss_xbar = ((sample - xbar) ** 2).sum()

    # tổng bình phương sai so với true mean: sum((xi - true_mean)^2)
    ss_true = ((sample - true_mean) ** 2).sum()

    # Các ước lượng phương sai (variance)
    s1_var = ss_xbar / (n - 1)  # unbiased estimator (chia cho n-1)
    s2_var = ss_xbar / n        # MLE (chia cho n) — biased for small n
    s3_var = ss_true / n        # khi biết true mean, chia cho n là hợp lý

    # lưu căn bậc hai (standard deviation) của từng ước lượng
    s1_sds[i] = sqrt(s1_var)
    s2_sds[i] = sqrt(s2_var)
    s3_sds[i] = sqrt(s3_var)

    # -------------------------
    # In ví dụ mẫu (để thấy một sample cụ thể)
    # -------------------------
    # Tạo lại RNG với cùng seed_sampling để lấy sample ví dụ giống lần rút đầu
    rng2 = np.random.default_rng(seed_sampling)
    example = rng2.choice(pop_int, size=n, replace=False)
    ex_xbar = example.mean()
    ex_ss_xbar = ((example - ex_xbar) ** 2).sum()
    ex_ss_true = ((example - true_mean) ** 2).sum()

print('\nExample sample (n=4):', example)
print(' sample mean:', ex_xbar)
print(' s1 sd (denom n-1):', sqrt(ex_ss_xbar / (n - 1)))
print(' s2 sd (denom n):  ', sqrt(ex_ss_xbar / n))
print(' s3 sd (true mean):', sqrt(ex_ss_true / n))

# -------------------------
# Kết quả trung bình qua MC
# -------------------------
print('\nAveraged over', MC, 'samples (std of estimators):')
print(' mean s1_sd =', np.mean(s1_sds))
print(' mean s2_sd =', np.mean(s2_sds))
print(' mean s3_sd =', np.mean(s3_sds))

print('\nAlso means of variances:')
print(' mean s1_var =', np.mean(s1_sds ** 2))
print(' mean s2_var =', np.mean(s2_sds ** 2))
print(' mean s3_var =', np.mean(s3_sds ** 2))

# -------------------------
# Lưu kết quả chi tiết của từng lần lặp vào CSV
# Mỗi dòng: s1_sd, s2_sd, s3_sd
# -------------------------
out = Path(__file__).with_name('sample_integers_results.csv')
np.savetxt(out, np.vstack([s1_sds, s2_sds, s3_sds]).T, delimiter=',', header='s1_sd,s2_sd,s3_sd', comments='')
print('\nSaved per-run stds to', out)
