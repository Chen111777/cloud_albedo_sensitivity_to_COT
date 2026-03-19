import miepython
import numpy as np
import matplotlib.pyplot as plt

# Parameters
d = 10.0          # 液滴直径 in µm（miepython 始终使用直径，非半径）
m_real = 1.33     # 水在可见光波段的折射率实部
m_imag = 0.0      # 可见光/近红外短波波段虚部≈0，无吸收
m = complex(m_real, m_imag) / 1.0  

wavelengths = np.arange(0.3, 5, 0.01)
g_values = []
ssa_values = []

for lam in wavelengths:
    x = (np.pi * d) / lam
    qext, qsca, qback, g = miepython.mie(m, x)
    g_values.append(g)
    ssa_values.append(qsca / qext)

# 绘图设置
plt.figure(figsize=(5,4))
plt.plot(wavelengths, g_values, color='k')
# plt.axhline(0.85, linestyle='--', label='g = 0.85')
plt.xlim([0,5])
plt.xlabel('$λ$ (µm)', fontsize=13)
plt.ylabel('$g$', fontsize=13)
plt.grid(True, alpha=0.3)
plt.show()
