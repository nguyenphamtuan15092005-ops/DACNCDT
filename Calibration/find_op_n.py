"""
Utilities for experiment: compute standard deviation (SD).

This module provides a small, dependency-free implementation of standard
deviation with configurable degrees of freedom (ddof). Use ddof=1 for the
sample standard deviation (the common choice when estimating from a sample),
or ddof=0 for the population standard deviation.

Example:
	>>> sd([1,2,3,4], ddof=0)  # population SD
	1.118033988749895
	>>> sd([1,2,3,4], ddof=1)  # sample SD
	1.2909944487358056
"""

from typing import Iterable
import math
from pathlib import Path


def load_values(path: str):
	"""Load one numeric value per line from a text file and return a list of floats."""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Input file not found: {p}")
	vals = []
	with p.open('r', encoding='utf-8') as f:
		for lineno, line in enumerate(f, start=1):
			s = line.strip()
			if not s:
				continue
			try:
				vals.append(float(s))
			except Exception:
				# skip non-numeric lines but warn
				print(f"warning: skipping non-numeric line {lineno}: {s}")
	return vals


def sd(values: Iterable[float], ddof: int = 1) -> float:
	"""
	Compute the standard deviation of an iterable of numbers.

	Args:
		values: Iterable of numeric values (ints/floats).
		ddof: Delta degrees of freedom. Use 0 for population SD, 1 for sample SD.

	Returns:
		The standard deviation as a float.

	Raises:
		ValueError: if no values are provided or if ddof >= n.
	"""
	vals = list(values)
	n = len(vals)
	if n == 0:
		raise ValueError("sd() requires at least one data point")
	if ddof >= n:
		raise ValueError("ddof must be smaller than the number of data points")

	# Use a numerically-stable two-pass algorithm: first compute the mean,
	# then the sum of squared deviations.
	mean_val = sum(vals) / n
	ssq = 0.0
	for x in vals:
		d = x - mean_val
		ssq += d * d

	variance = ssq / (n - ddof)
	return math.sqrt(variance)


if __name__ == "__main__":
	# If run directly, try to read data_10cm_3rd.txt in the same folder and print stats
	default_path = Path(__file__).with_name('211cm_100.txt')
	try:
		values = load_values(str(default_path))
	except FileNotFoundError:
		print(f"Default data file not found: {default_path}\nPlease create or place `data_10cm_3rd.txt` next to this script.")
		# fall back to demo
		values = [1, 2, 3, 4]

	print(f"Loaded {len(values)} values")
	if len(values) == 0:
		print("No numeric values to analyse. Exiting.")
	else:
		mu = sum(values) / len(values)
		sd_sample = sd(values, ddof=1) if len(values) > 1 else 0.0
		sd_pop = sd(values, ddof=0) if len(values) > 0 else 0.0
		mn = min(values)
		mx = max(values)
		sem = sd_sample / math.sqrt(len(values)) if len(values) > 0 else 0.0

		print(f"mean (mu): {mu:.6f}")
		print(f"SD (sample, ddof=1): {sd_sample:.6f}")
		print(f"SD (population, ddof=0): {sd_pop:.6f}")
		print(f"min: {mn:.6f}")
		print(f"max: {mx:.6f}")
		print(f"SEM = SD_sample / sqrt(n): {sem:.6f}")

		# --- compute SD and SEM as functions of n (x axis 1..N) and plot ---
		try:
			import matplotlib.pyplot as plt
		except Exception:
			print("matplotlib not installed. To enable plotting, install with: pip install matplotlib")
		else:
			N = len(values)
			ns = list(range(1, N + 1))
			sd_n = []
			sem_n = []
			for n in ns:
				if n <= 1:
					sd_n.append(0.0)
					sem_n.append(0.0)
				else:
					# use our sd() function for sample SD (ddof=1)
					cur_sd = sd(values[:n], ddof=1)
					sd_n.append(cur_sd)
					sem_n.append(cur_sd / math.sqrt(n))

			# find smallest n>=2 where SEM <= 0.02 (if any)
			selected_n = None
			sem_check = 0.01
			for n, sem_val in zip(ns, sem_n):
				print(sem_val)
				if n > 15 and sem_val < sem_check:
					selected_n = n - 1
					print(sem_val)
					break

			if selected_n is not None:
				# compute SD and SEM at selected n for printing and annotation
				sd_sel = sd(values[:selected_n], ddof=1)
				sem_sel = sd_sel / math.sqrt(selected_n)
				print(f"Selected n = {selected_n}, SD = {sd_sel:.6f}, SEM = {sem_sel:.6f}")

			plt.figure(figsize=(8, 4.5))
			plt.plot(ns, sd_n, label='SD (sample, ddof=1)')
			plt.plot(ns, sem_n, label='SEM = SD/sqrt(n)')
			# add purple 'x' markers at each n for both series
			plt.scatter(ns, sd_n, marker='x', color='purple', s=40, label='_nolegend_')
			plt.scatter(ns, sem_n, marker='x', color='purple', s=30, label='_nolegend_')
			if selected_n is not None:
				plt.axvline(selected_n, color='red', linestyle='--', label=f'selected n={selected_n}')
				# add a text annotation near the line
				plt.text(selected_n + 1, max(sd_n[selected_n-1], sem_n[selected_n-1]) * 1.05,
					f"n={selected_n}\nSD={sd_sel:.3f}\nSEM={sem_sel:.3f}",
					color='red', fontsize=9, va='bottom')
			plt.xlim(1, max(1, N))
			plt.xlabel('n (number of measurements)')
			plt.ylabel('value')
			plt.title('SD and SEM vs Distance 211 cm')
			plt.grid(True)
			plt.legend()
			plt.tight_layout()
			out_png = Path(__file__).with_name('data_n_cm_sd_sem.png')
			plt.savefig(out_png)
			print(f"Plot saved to: {out_png}")
			# also show the plot window
			plt.show()