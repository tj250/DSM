import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
from scipy.optimize import minimize
import pandas
from DSMAlgorithms import DataDistributionType

'''
对经过Yeo-Johnson变换后的数据进行逆变换
'''


def inverse_yeojohnson(data, lmbda):
    if data >= 0:
        if lmbda == 0:
            return np.exp(data) - 1
        else:
            return (data * lmbda + 1) ** (1 / lmbda) - 1
    else:
        if lmbda == 2:
            return 1 - np.exp(-data)
        else:
            return 1 - (-data * (2 - lmbda) + 1) ** (1 / (2 - lmbda))


'''
平方根反正弦变换‌（Arcsine Square Root Transformation）
适用场景‌：百分比或比率数据（如二项分布数据）
输入值需在[0,1]范围内（百分比需先除以100），否则会报错
'''


def arcsine_sqrt_array(x):
    return np.arcsin(np.sqrt(x))


'''
连续型数据分布通常包括：高斯（正态）分布、gamma分布、指数分布、beta分布、均匀（Uniform ）分布

在Python中可以通过多种统计检验方法来判断数据是否符合正态分布。以下是常用的几种方法实现：

Shapiro-Wilk检验（适合小样本数据）
Kolmogorov-Smirnov检验
D'Agostino's K²检验
安德森-达令检验
结合Q-Q图可视化判断

这段代码提供了四种统计检验方法和可视化分析，可以综合判断数据是否符合正态分布。使用时需要注意：
1) 不同检验方法对样本量敏感度不同；
2) p值大于显著性水平(默认0.05)时不能拒绝原假设(数据来自正态分布)；
3) 建议结合统计检验和可视化结果综合判断。

样本量驱动选择:
n ≤ 50：优先用 Shapiro-Wilk
50 < n ≤ 1000：推荐 Anderson-Darling
n > 1000：‌D'Agostino's K²‌ 或结合Q-Q图


如果通过多种转换方法依然无法达到较为理想的效果，此时最好放弃使用基于正态性分布为前提的统计学分析方法(如t检验、方差分析)，
可选择适合非正态分布数据的方法(如非参数检验)进行数据分析。
'''


def normality_distribution_test(data, p=0.05):
    """
    综合正态性检验函数
    :param data: 待检验数据
    :param p: 显著性水平
    :return: 是否符合正态分布
    """
    results = {}

    # Shapiro-Wilk检验:基于数据排序值与理论值的相关性，对小样本敏感。
    shapiro_stat, shapiro_p = stats.shapiro(data)
    results['Shapiro-Wilk'] = {
        'statistic': shapiro_stat,
        'p-value': shapiro_p,
        'normal': float(shapiro_p) > p
    }

    # Kolmogorov-Smirnov检验:比较样本累积分布与理论分布的差异
    ks_stat, ks_p = stats.kstest(data, 'norm')
    results['Kolmogorov-Smirnov'] = {
        'statistic': ks_stat,
        'p-value': ks_p,
        'normal': float(ks_p) > p
    }

    # D'Agostino's K²检验:联合检验偏度（对称性）和峰度（陡峭度）是否偏离正态。
    k2_stat, k2_p = stats.normaltest(data)
    results['D\'Agostino'] = {
        'statistic': k2_stat,
        'p-value': k2_p,
        'normal': float(k2_p) > p
    }

    # Anderson-Darling检验:加强KS检验对尾部数据的敏感性。
    anderson_result = stats.anderson(data, dist='norm')
    results['Anderson-Darling'] = {
        'statistic': anderson_result.statistic,
        'critical_values': anderson_result.critical_values,
        'significance_level': anderson_result.significance_level,
        'normal': all(anderson_result.statistic < cv for cv in anderson_result.critical_values)
    }
    if data.shape[0] <= 20:  # 小样本量
        return results['Shapiro-Wilk']['normal'] or results['Anderson-Darling']['normal']
    elif data.shape[0] > 20 and data.shape[0] <= 50:  # 小样本量
        return results['Shapiro-Wilk']['normal'] or results['Anderson-Darling']['normal'] or results['D\'Agostino'][
            'normal']
    elif data.shape[0] > 50 and data.shape[0] <= 100:  # 中等样本量
        return results['Shapiro-Wilk']['normal'] or results['Anderson-Darling']['normal'] or results['D\'Agostino'][
            'normal']
    elif data.shape[0] > 100 and data.shape[0] <= 1000:  # 中大样本量
        return results['Shapiro-Wilk']['normal'] or results['Anderson-Darling']['normal'] or results['D\'Agostino'][
            'normal']
    elif data.shape[0] > 1000 and data.shape[0] <= 5000:  # 大样本量
        return results['Shapiro-Wilk']['normal'] or results['Anderson-Darling']['normal'] or results['D\'Agostino'][
            'normal'] or results['Kolmogorov-Smirnov']['normal']
    else:  # >5000时
        return results['Kolmogorov-Smirnov']['normal']


'''
可视化正态分布数据
'''


def normality_visual(data):
    # 绘制Q-Q图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.subplot(1, 2, 2)
    sns.histplot(data, kde=True)
    plt.title('Distribution with KDE')

    plt.tight_layout()
    plt.show()


'''
检测数据经过多种变化后是否符合正态分布
'''


def normality_test_after_transormed(data, p_value=0.05, is_percent = False):
    """
    检测数据经过多种变化后是否符合正态分布
    :param data: 待检验数据
    :param p_value: 显著性水平
    :param is_percent: 是否为百分比数据（介于0-100%之间），如砾石含量
    :return: 检验结果字典
    """
    if np.any(data <= 0):  # 如果数据中包含负数或0
        transformed_data, lambda_val = yeojohnson(data)
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, float(lambda_val), 'Yeo-Johnson'  # Yeo-Johnson变换
    else:  # 数据中不包含负数或0
        transformed_data = boxcox(data, lmbda=0)
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, 0.0, 'logarithmic'  # 对数变换
        transformed_data = boxcox(data, lmbda=0.5)
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, 0.5, 'square_root'  # 平方根变换
        transformed_data = boxcox(data, lmbda=1 / 3)
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, 1 / 3, 'cube_root'  # 立方根变换
        transformed_data = boxcox(data, lmbda=-1)
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, -1.0, 'reciprocal'  # 倒数变换
        try:
            transformed_data, lambda_val = boxcox(data)
        except Exception as ex:  # 可能会产生异常，例如：Data must not be constant.
            print(ex)
            return None, None, None
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, float(lambda_val), 'box-cox'  # box-cox变换
        transformed_data, lambda_val = yeojohnson(data)
        if normality_distribution_test(transformed_data, p_value):
            return transformed_data, float(lambda_val), 'Yeo-Johnson'  # Yeo-Johnson变换
        if is_percent:  # 是百分比数据,则可以尝试平方根反正弦变换
            transformed_data = arcsine_sqrt_array(data)
            if normality_distribution_test(transformed_data, p_value):
                return transformed_data, 0.0, 'arcsine_sqrt'  # 平方根反正弦变换
    return None, None, None  # 未找到合适的变换方法，使得数据符合正态分布


'''
对数据进行逆变换
'''


def inverse_data(data, transformed_type, lmbda):
    if transformed_type == 'Yeo-Johnson':
        return inverse_yeojohnson(data, lmbda)  # 调用自定义的逆变换方法实现
    else:
        return inv_boxcox(data, lmbda)



'''
gamma分布检验
'''
def gamma_distribution_test(data, p=0.05):
    """
    全自动Gamma分布检验
    :param data: 输入数据数组（需为正数）
    :param p: 显著性水平，默认0.05
    :return: (bool) True表示服从Gamma分布，False表示不服从
    """
    # 1. 数据预处理（移除非正数）
    data = np.array(data)
    data = data[data > 0]
    if len(data) < 10:
        return False  # 样本量不足直接返回False

    # 2. 参数估计（MLE方法）
    try:
        shape, _, scale = stats.gamma.fit(data, floc=0)
    except:
        return False  # 拟合失败时返回False

    # 3. K-S检验（自动判断）
    try:
        _, p_value = stats.kstest(data, 'gamma', args=(shape, 0, scale))
        return p_value > p
    except:
        return False




'''
逆高斯分布检测
'''
def inverse_gaussian_distribution_test(data, p=0.05):
    """
    自动化检测数据是否符合逆高斯分布

    参数:
    data: 待检测的数据数组
    alpha: 显著性水平(默认0.05)

    返回:
    bool: True表示数据符合逆高斯分布，False表示不符合
    """
    # 1. 数据基本检查
    if len(data) < 50:
        # raise ValueError("样本量太小(至少需要50个数据点)")
        return False

    # 确保数据都是正数(逆高斯分布要求)
    if np.any(data <= 0):
        return False

    # 2. 使用最大似然估计(MLE)获取参数
    def neg_log_likelihood(params):
        mu, lam = params
        if mu <= 0 or lam <= 0:
            return np.inf
        n = len(data)
        term1 = (n / 2) * np.log(lam / (2 * np.pi))
        term2 = (3 / 2) * np.sum(np.log(data))
        term3 = (lam / (2 * mu ** 2)) * np.sum((data - mu) ** 2 / data)
        return - (term1 - term2 - term3)

    # 使用矩估计作为初始值
    mu0 = np.mean(data)
    lam0 = mu0 ** 3 / np.var(data)

    # 优化最大似然函数
    result = minimize(neg_log_likelihood, [mu0, lam0], method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)])

    if not result.success:
        # 如果优化失败，使用矩估计
        mu_hat, lam_hat = mu0, lam0
    else:
        mu_hat, lam_hat = result.x

    # 3. 使用自定义的K-S检验
    # 排序数据
    sorted_data = np.sort(data)

    # 计算经验累积分布函数
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n

    # 计算理论累积分布函数
    # 使用scipy的invgauss分布，但注意参数化方式
    # scipy的invgauss使用mu = mean * shape，其中shape = 1/lambda
    # 所以我们需要调整参数
    shape_param = mu_hat / lam_hat
    scale_param = lam_hat
    theoretical_cdf = stats.invgauss.cdf(sorted_data, shape_param, scale=scale_param)

    # 计算K-S统计量
    D = np.max(np.abs(ecdf - theoretical_cdf))

    # 4. 计算p值（使用近似公式）
    # K-S检验的p值近似公式
    p_value = 2 * np.exp(-2 * (D * np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) ** 2)

    # 5. 返回结果
    return p_value > p

'''
泊松分布检测
'''
def poisson_distribution_test(data, p=0.05):
    """
    自动化检测数据是否符合泊松分布

    参数:
    data: 待检测的数据数组（应为非负整数）
    alpha: 显著性水平(默认0.05)

    返回:
    bool: True表示数据符合泊松分布，False表示不符合
    """
    # 1. 基本数据检查
    if len(data) < 30:
        raise ValueError("样本量太小(至少需要30个数据点)")

    # 检查数据是否为整数（泊松分布要求）
    if not np.all(np.equal(np.mod(data, 1), 0)):
        return False

    # 检查数据是否非负（泊松分布要求）
    if np.any(data < 0):
        return False

    # 2. 计算均值和方差
    mean_val = np.mean(data)
    var_val = np.var(data, ddof=0)  # 使用总体方差

    # 3. 泊松分布的均值和方差应该大致相等
    # 使用变异系数(方差/均值)进行初步判断
    if mean_val == 0:  # 防止除以零
        dispersion_ratio = 0
    else:
        dispersion_ratio = var_val / mean_val

    # 如果变异系数明显偏离1，可能不是泊松分布
    if dispersion_ratio < 0.8 or dispersion_ratio > 1.2:
        return False

    # 4. 卡方拟合优度检验
    # 使用最大似然估计泊松参数λ
    lambda_est = mean_val

    # 获取唯一值和频数
    unique_vals, observed_counts = np.unique(data, return_counts=True)
    n_categories = len(unique_vals)
    n_total = len(data)

    # 计算期望概率和频数
    # 确保覆盖所有观察到的值
    max_val = int(np.max(data))
    expected_probs = stats.poisson.pmf(range(max_val + 1), lambda_est)

    # 确保概率总和为1（处理浮点精度问题）
    expected_probs = expected_probs / np.sum(expected_probs)

    # 创建完整的观察频数数组（包括可能缺失的值）
    full_observed = np.zeros(max_val + 1)
    for val, count in zip(unique_vals, observed_counts):
        if val <= max_val:
            full_observed[int(val)] = count

    # 合并低期望频数的类别（确保期望频数不小于5）
    min_expected = 5
    adjusted_observed = []
    adjusted_expected = []

    current_obs = 0
    current_exp = 0
    i = 0

    while i <= max_val:
        current_obs += full_observed[i]
        current_exp += n_total * expected_probs[i]

        # 如果当前期望频数足够大，或者是最后一个值
        if current_exp >= min_expected or i == max_val:
            adjusted_observed.append(current_obs)
            adjusted_expected.append(current_exp)
            current_obs = 0
            current_exp = 0

        i += 1

    # 如果合并后类别太少，检验可能不可靠
    if len(adjusted_observed) < 3:
        return False

    # 确保观察频数和期望频数的总和匹配
    total_observed = sum(adjusted_observed)
    total_expected = sum(adjusted_expected)

    # 如果总和差异较大，调整期望频数
    if abs(total_observed - total_expected) > 1e-8:
        scale_factor = total_observed / total_expected
        adjusted_expected = [exp * scale_factor for exp in adjusted_expected]

    # 执行卡方检验
    try:
        chi2, p_value = stats.chisquare(adjusted_observed, adjusted_expected, ddof=1)
    except ValueError:
        # 如果卡方检验失败，返回False
        return False

    # 5. 返回结果(p值大于显著性水平则接受原假设)
    return p_value > p


'''
复合泊松-伽马分布检测
'''
def compound_poisson_gamma_distribution_test(data, threshold=0.8):
    """
    检测数据是否符合复合泊松-伽马分布

    参数:
    data: 待检测的数据数组
    threshold: 匹配阈值(默认0.8)

    返回:
    bool: True表示数据符合复合泊松-伽马分布，False表示不符合
    """
    # 1. 数据基本检查
    if len(data) < 100:
        return False  # 样本量太小，无法可靠检测

    # 确保数据都是非负数
    if np.any(data < 0):
        return False

    # 2. 计算基本统计量
    mean_val = np.mean(data)
    var_val = np.var(data)
    zero_proportion = np.sum(data == 0) / len(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # 3. 复合泊松-伽马分布的特征检查
    # 检查方差-均值关系：对于复合泊松-伽马分布，方差应该大于均值
    if var_val <= mean_val:
        return False

    # 检查偏度：复合泊松-伽马分布应该是正偏的
    if skewness <= 0:
        return False

    # 检查峰度：复合泊松-伽马分布应该有较高的峰度
    if kurtosis <= 0:
        return False

    # 4. 计算复合泊松-伽马分布的特征得分
    score = 0
    max_score = 5  # 最大可能得分

    # 特征1: 方差大于均值
    if var_val > mean_val:
        score += 1

    # 特征2: 正偏态
    if skewness > 0:
        score += 1

    # 特征3: 高峰度
    if kurtosis > 0:
        score += 1

    # 特征4: 适当的零值比例
    # 复合泊松-伽马分布的零值比例通常在0.1到0.9之间
    if 0.1 <= zero_proportion <= 0.9:
        score += 1

    # 特征5: 方差与均值的关系
    # 对于复合泊松-伽马分布，log(Var) ≈ p * log(Mean)
    # 其中1 < p < 2
    if mean_val > 0 and var_val > 0:
        p_est = np.log(var_val) / np.log(mean_val)
        if 1.1 <= p_est <= 1.9:
            score += 1

    # 5. 返回结果
    return score / max_score >= threshold

'''
检测数据的分布
'''
def check_data_distribution(data, p_value=0.05)->DataDistributionType:
    if normality_distribution_test(data, p_value):
        return DataDistributionType.Normal
    if gamma_distribution_test(data, p_value):
        return DataDistributionType.Gamma
    if poisson_distribution_test(data, p_value):
        return DataDistributionType.Possion
    if inverse_gaussian_distribution_test(data, p_value):
        return DataDistributionType.InverseGaussin
    if compound_poisson_gamma_distribution_test(data):
        return DataDistributionType.CompoundPossionGamma
    return DataDistributionType.Unknown

# 示例使用
if __name__ == "__main__":
    # 生成正态分布数据
    # normal_data = np.random.normal(loc=0, scale=1, size=1000)
    # 生成非正态分布数据
    # non_normal_data = np.random.exponential(scale=1, size=1000)

    normal_data = pandas.read_csv(r'D:\PycharmProjects\DSM\pHEnv2-full.csv')['pH'].values  # csv中包含了坐标点的列：coordinates

    print("正态数据检验结果:")
    print(normality_distribution_test(normal_data))
    print(normality_test_after_transormed(normal_data))
    # normality_visual(normal_data)

    print("\n非正态数据检验结果:")
    # print(normality_test(non_normal_data))


    # ---------如下进行gamma分布检验的测试---------
    # 测试数据生成（实际使用时替换为真实数据）
    gamma_data = stats.gamma.rvs(2, scale=1, size=1000)
    normal_data = np.random.normal(2, 1, 1000)

    # 自动化检验
    print("Gamma数据检验结果:", gamma_distribution_test(gamma_data))  # 应输出True
    print("正态数据检验结果:", gamma_distribution_test(normal_data))  # 应输出False

    # ---------如下进行逆高斯分布分布检验的测试---------
    np.random.seed(42)

    # 生成逆高斯分布数据
    # 注意：scipy的invgauss参数化方式与标准不同
    # scipy: invgauss(mu, scale=1) 其中mu = mean/shape, shape = 1/lambda
    mu_true, lambda_true = 2.0, 3.0
    shape_param = mu_true / lambda_true
    scale_param = lambda_true

    print(f"真实参数: μ={mu_true}, λ={lambda_true}")
    print(f"Scipy参数: shape={shape_param}, scale={scale_param}")

    inv_gaussian_data = stats.invgauss(shape_param, scale=scale_param).rvs(size=1000)

    # 生成正态分布数据作为对比
    normal_data = np.random.normal(5, 1, 1000)

    # 检测数据
    result_ig = inverse_gaussian_distribution_test(inv_gaussian_data)
    result_normal = inverse_gaussian_distribution_test(normal_data)

    print("逆高斯分布数据检测结果:", result_ig)
    print("正态分布数据检测结果:", result_normal)

    # 打印数据统计信息
    print("\n逆高斯分布数据统计信息:")
    print(f"均值: {np.mean(inv_gaussian_data):.3f}, 方差: {np.var(inv_gaussian_data):.3f}")
    print(f"偏度: {stats.skew(inv_gaussian_data):.3f}, 峰度: {stats.kurtosis(inv_gaussian_data):.3f}")

    print("\n正态分布数据统计信息:")
    print(f"均值: {np.mean(normal_data):.3f}, 方差: {np.var(normal_data):.3f}")
    print(f"偏度: {stats.skew(normal_data):.3f}, 峰度: {stats.kurtosis(normal_data):.3f}")

    # ----------------以下验证数据是否符合泊松分布--------------
    np.random.seed(42)

    # 生成泊松分布数据(应该返回True)
    lambda_true = 5.0
    poisson_data = np.random.poisson(lambda_true, 1000)

    # 生成均匀分布数据(应该返回False)
    uniform_data = np.random.randint(0, 10, 1000)

    # 检测数据
    print("泊松分布数据检测结果:", poisson_distribution_test(poisson_data))
    print("均匀分布数据检测结果:", poisson_distribution_test(uniform_data))

    # 打印一些统计信息以帮助理解
    print("\n泊松分布数据统计信息:")
    print(f"均值: {np.mean(poisson_data):.3f}, 方差: {np.var(poisson_data):.3f}")
    print(f"变异系数(方差/均值): {np.var(poisson_data) / np.mean(poisson_data):.3f}")

    print("\n均匀分布数据统计信息:")
    print(f"均值: {np.mean(uniform_data):.3f}, 方差: {np.var(uniform_data):.3f}")
    print(f"变异系数(方差/均值): {np.var(uniform_data) / np.mean(uniform_data):.3f}")

    # --------------以下验证数据是否符合复合泊松-伽马分布----------------
    print('\n\n--------------以下验证数据是否符合复合泊松-伽马分布----------------')

    # 生成示例数据
    np.random.seed(42)

    # 生成复合泊松-伽马分布数据
    n = 1000
    # 使用更简单的方法生成复合泊松-伽马分布数据
    # 首先生成泊松计数
    lambda_param = 2.0
    poisson_counts = np.random.poisson(lambda_param, n)

    # 然后为每个计数生成伽马分布的和
    alpha_param = 2.0  # 伽马形状参数
    beta_param = 1.0  # 伽马尺度参数

    compound_data = np.zeros(n)
    for i in range(n):
        if poisson_counts[i] > 0:
            compound_data[i] = np.sum(np.random.gamma(alpha_param, beta_param, poisson_counts[i]))

    # 生成正态分布数据作为对比
    normal_data = np.random.normal(5, 2, n)
    # 确保正态分布数据非负
    normal_data = np.clip(normal_data, 0, None)

    # 检测数据
    result_compound = compound_poisson_gamma_distribution_test(compound_data)
    result_normal = compound_poisson_gamma_distribution_test(normal_data)

    print("复合泊松-伽马分布数据检测结果:", result_compound)
    print("正态分布数据检测结果:", result_normal)

    # 打印一些统计信息以帮助理解
    print("\n复合泊松-伽马分布数据统计信息:")
    print(f"均值: {np.mean(compound_data):.3f}, 方差: {np.var(compound_data):.3f}")
    print(f"零值比例: {np.sum(compound_data == 0) / len(compound_data):.3f}")
    print(f"偏度: {stats.skew(compound_data):.3f}, 峰度: {stats.kurtosis(compound_data):.3f}")

    print("\n正态分布数据统计信息:")
    print(f"均值: {np.mean(normal_data):.3f}, 方差: {np.var(normal_data):.3f}")
    print(f"零值比例: {np.sum(normal_data == 0) / len(normal_data):.3f}")
    print(f"偏度: {stats.skew(normal_data):.3f}, 峰度: {stats.kurtosis(normal_data):.3f}")