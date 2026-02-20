import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats

# 设置画布
plt.figure(figsize=(15, 12), dpi=100)
plt.suptitle('常见连续型概率分布图示', fontsize=18)

# 1. 正态分布（Gaussian）
ax1 = plt.subplot(231)
x = np.linspace(-5, 5, 1000)
y_norm = stats.norm.pdf(x)
plt.plot(x, y_norm, 'b-', lw=2)
plt.fill_between(x, y_norm, where=(x>1) & (x<2), alpha=0.3, color='skyblue')
plt.title('正态分布\n(Normal Distribution)')
plt.grid(True)

# 2. 指数分布（Exponential）
ax2 = plt.subplot(232)
x = np.linspace(0, 5, 1000)
y_exp = stats.expon.pdf(x)
plt.plot(x, y_exp, 'r-', lw=2)
plt.title('指数分布\n(Exponential Distribution)')
plt.grid(True)

# 3. 均匀分布（Uniform）
ax3 = plt.subplot(233)
x = np.linspace(-2, 4, 1000)
y_uni = stats.uniform.pdf(x, loc=0, scale=3)
plt.plot(x, y_uni, 'g-', lw=2)
plt.title('均匀分布\n(Uniform Distribution)')
plt.grid(True)

# 4. 伽马分布（Gamma）
ax4 = plt.subplot(234)
x = np.linspace(0, 10, 1000)
y_gamma1 = stats.gamma.pdf(x, a=1)  # α=1 (指数分布)
y_gamma2 = stats.gamma.pdf(x, a=2)  # α=2
y_gamma3 = stats.gamma.pdf(x, a=4)  # α=4
plt.plot(x, y_gamma1, 'b-', label='α=1')
plt.plot(x, y_gamma2, 'r--', label='α=2')
plt.plot(x, y_gamma3, 'g-.', label='α=4')
plt.title('伽马分布\n(Gamma Distribution)')
plt.legend()
plt.grid(True)

# 5. 贝塔分布（Beta）
ax5 = plt.subplot(235)
x = np.linspace(0, 1, 1000)
y_beta1 = stats.beta.pdf(x, a=0.5, b=0.5)  # U型
y_beta2 = stats.beta.pdf(x, a=2, b=2)     # 钟型
y_beta3 = stats.beta.pdf(x, a=5, b=1)     # J型
plt.plot(x, y_beta1, 'b-', label='α=0.5, β=0.5')
plt.plot(x, y_beta2, 'r--', label='α=2, β=2')
plt.plot(x, y_beta3, 'g-.', label='α=5, β=1')
plt.title('贝塔分布\n(Beta Distribution)')
plt.legend()
plt.grid(True)

# 6. 对数正态分布（Log-normal）
ax6 = plt.subplot(236)
x = np.linspace(0, 5, 1000)
y_lognorm = stats.lognorm.pdf(x, s=0.5)  # s=σ
plt.plot(x, y_lognorm, 'm-', lw=2)
plt.title('对数正态分布\n(Log-normal Distribution)')
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('continuous_distributions.png', bbox_inches='tight')
plt.show()
