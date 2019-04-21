import numpy as np
import matplotlib.pyplot as plt

# 学習データを行列として読み込む
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)

# 1列目の抽出
train_x = train[:,0]
# 2列目の抽出
train_y = train[:,1]

# 標準化
def standardize(x):
    mu = train_x.mean()
    sigma = train_x.std()
    return (x - mu) / sigma

# 1列目の標準化
train_z = standardize(train_x)

# 学習率
ETA = 1e-3

# 平均二乗誤差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

# パラメータをランダムに初期化
theta = np.random.rand(3)

# 平均二乗誤差の履歴
errors = []

# 学習データの[1, x, x^2]の行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 予測関数
# f_theta(x) = X.theta
def f(x):
    return np.dot(x, theta)

X = to_matrix(train_z)

# 誤差の差分
diff = 1

# 学習を繰り返す
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # パラメータを更新
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 前回の誤差との差分を計算
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

# 誤差をプロット
x = np.arange(len(errors))

plt.plot(x, errors)
plt.show()