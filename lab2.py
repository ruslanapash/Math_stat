import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, norm

file_path = 'data.xlsx'
gamma = 0.95
alpha = 1 - gamma

# Завантаження та перевірка
try:
    raw = pd.read_excel(file_path, header=None, index_col=0)
    df = raw.T.reset_index(drop=True)
    df.columns = ["interval", "n_i"]
except FileNotFoundError:
    print(f"Помилка: Файл '{file_path}' не знайдено.")
    sys.exit()

n_i = df["n_i"].astype(int).to_numpy()
n = n_i.sum()
if n == 99:
    print("\nСума частот n_i = 99.")
else:
    print("\nСума частот n_i не дорівнює 99.")
    sys.exit()

bounds = df["interval"].str.split(r"[-–]", expand=True, regex=True).astype(float)
a = bounds[0].to_numpy()
b = bounds[1].to_numpy()
h = b[0] - a[0]
x = (a + b) / 2
print("\nСередні значення інтервалів (x):", x)

# Точкові оцінки
print("\nОбчислення середнього значення (x̄):")
print("x_i * n_i:")
for x_k, n_k in zip(x, n_i):
    print(f"{x_k * n_k};")
sum_xn = np.sum(x * n_i)
print("Сума x_i * n_i:", sum_xn)
x_bar = sum_xn / n
print(f"Середнє значення (x̄): {x_bar:.4f}")

print("\nx_i - x̄:")
for i in range(len(x)):
    print(f"{x[i] - x_bar:.4f};")

print("\nn_i * (x_i - x̄)^2:")
for i in range(len(x)):
    print(f"{n_i[i] * (x[i] - x_bar) ** 2:.4f};")
ss = np.sum(n_i * (x - x_bar) ** 2)
print(f"Сума n_i * (x_i - x̄)^2: {ss:.4f}")

d = ss / n
s2 = ss / (n - 1)
s = np.sqrt(s2)
print(f"\nВибіркова дисперсія (d): {d:.4f}")
print(f"Виправлена дисперсія (s²): {s2:.4f}")
print(f"Виправлене середнє квадратичне відхилення (s): {s:.4f}")
print(f"Контроль: s²/d = {s2 / d:.4f} = {n / (n - 1):.4f} = {n}/{n - 1}")

# Довірчий інтервал для m
k = n - 1
t_gamma = t.ppf(1 - alpha / 2, k)
se = s / np.sqrt(n)
delta = t_gamma * se
m_low, m_high = x_bar - delta, x_bar + delta
print(f"\nk = {k}, t_γ = {t_gamma:.4f}")
print(f"Стандартна похибка s/√n = {se:.4f}")
print(f"Точність δ = {delta:.4f}")
print(f"{m_low:.4f} < m < {m_high:.4f}")

# Довірчі інтервали для σ² та σ 
chi2_1 = chi2.ppf(1 - alpha / 2, k)   # більша критична точка
chi2_2 = chi2.ppf(alpha / 2, k)       # менша критична точка
print(f"\nχ²_1 = {chi2_1:.3f}, χ²_2 = {chi2_2:.3f}")

var_low = (n - 1) * s2 / chi2_1       # більша точка у знаменнику лівої межі
var_high = (n - 1) * s2 / chi2_2
print(f"{var_low:.4f} < σ² < {var_high:.4f}")
print(f"{np.sqrt(var_low):.4f} < σ < {np.sqrt(var_high):.4f}")

# Відносна точність, ширини, асиметрія
epsilon = delta / x_bar * 100
print(f"\nВідносна точність ε = {epsilon:.2f} %")
print(f"Ширина інтервалу для m: {2 * delta:.4f}")
print(f"Ширина інтервалу для σ²: {var_high - var_low:.4f}")
print(f"Ширина інтервалу для σ: {np.sqrt(var_high) - np.sqrt(var_low):.4f}")
left_arm = s2 - var_low
right_arm = var_high - s2
print(f"Ліве плече: {left_arm:.4f}, праве плече: {right_arm:.4f}, "
      f"праве/ліве = {right_arm / left_arm:.2f}")

# Вплив надійності, t проти нормального
print("\nВплив надійності:")
print(f"{'γ':>5} {'t':>8} {'δ':>8} {'інтервал для m':>20} {'z':>8} {'δ_z':>8} {'різн.,%':>8}")
for g in (0.90, 0.95, 0.99):
    al = 1 - g
    t_g = t.ppf(1 - al / 2, k)
    z_g = norm.ppf(1 - al / 2)
    dl = t_g * se
    dl_z = z_g * se
    diff = (dl - dl_z) / dl * 100
    print(f"{g:>5} {t_g:>8.4f} {dl:>8.4f} "
          f"({x_bar - dl:.4f}; {x_bar + dl:.4f}) {z_g:>8.4f} {dl_z:>8.4f} {diff:>8.2f}")

# Потрібний обсяг вибірки
n_new = 4 * n
delta_approx = delta / 2
delta_exact = t.ppf(1 - alpha / 2, n_new - 1) * s / np.sqrt(n_new)
print(f"\nn' = 4 * n = {n_new}")
print(f"δ' ≈ δ/2 = {delta_approx:.4f}")
print(f"δ' з урахуванням зміни t-квантиля = {delta_exact:.4f}")

# Гістограма та полігон частот
plt.bar(x, n_i, width=h, color='lightblue', edgecolor='black', label='гістограма')
plt.plot(x, n_i, marker='o', color='red', label='полігон частот')
plt.axvline(x_bar, color='black', linestyle='--', label=f'x̄ = {x_bar:.4f}')
plt.xlabel('Період власних коливань, мс')
plt.ylabel('Частота n_i')
plt.title('Гістограма та полігон частот')
plt.legend()
plt.show()