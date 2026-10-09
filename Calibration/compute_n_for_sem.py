from pathlib import Path
import math
import statistics

'''
Bước 1: Chọn một giá trị thực bất kỳ trong giới hạn đo của cảm biến
Bước 2: Tiến hành đo 100 lần
Bước 3: Tính các thông số sau
Giá trị trung bình quần thể (μ)
Độ lệch chuẩn (SD)
Giá trị nhỏ nhất (min)
Giá trị lớn nhất (max)
Bước 4: Lập mối quan hệ giữa sai số chuẩn trung bình (SEM) và số phép đo trong một
mẫu (n)
Bước 5: Chọn số lần đo cần thiết
cach lap trinh va chuyen du lieu tu file excel
'''

p = Path(__file__).with_name('data_64cm_3rd.txt')
if not p.exists():
    print(f"Data file not found: {p}")
    raise SystemExit(1)

vals = [float(line.strip()) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
N = len(vals)
threshold = 0.01
found = None
# start from n=2 because sample standard deviation requires at least 2 samples
for n in range(2, N+1):
    sd = statistics.stdev(vals[:n])
    #print(round(sd,6))
    sem = sd / math.sqrt(n)
    if sem <= threshold and n > 15 and n <= 50:
        found = (n, sem)
        #break
    print(round(sem,6))

if found:
    n_sel, sem_sel = found
    # compute SD at selected n
    sd_sel = statistics.stdev(vals[:n_sel]) if n_sel > 1 else 0.0
    print(f"n = {n_sel}, SD = {sd_sel:.6f}, SEM = {sem_sel:.6f} <= {threshold}")
else:
    sdN = statistics.stdev(vals) if N > 1 else 0.0
    semN = sdN / math.sqrt(N) if N > 0 else 0.0
    print(f"No n in 1..{N} has SEM <= {threshold}; SEM at N = {semN:.6f}")
