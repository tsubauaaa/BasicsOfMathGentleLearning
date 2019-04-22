import numpy as np
import matplotlib.pyplot as plt

# 学習データを行列として読み込む
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)

# 1列目を抽出して行ベクトルにする
train_x = train[:,0]
# 2列目を抽出して行ベクトルにする
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

# パラメータをランダムに初期化
theta = np.random.rand(3)

# 学習データの[1, x, x^2]の行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

# 予測関数
# f_theta(x) = X.theta
def f(x):
    return np.dot(x, theta)

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

X = to_matrix(train_z)

# 平均二乗誤差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

# 誤差の差分
diff = 1

# 平均二乗誤差の履歴
errors = []

# 学習を繰り返す
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # 学習データを並べ替えるためランダムな順列を用意する
    p = np.random.permutation(X.shape[0])
    # 学習データをランダムに取り出して確率的勾配降下法でパラメータを更新
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    # 前回の誤差との差分を計算
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

# 誤差をプロット
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()