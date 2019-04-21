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

# パラメータをランダムに初期化
theta = np.random.rand(3)

# 学習データの[1, x, x^2]の行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)

# 予測関数
# f^T.
def f(x):
    return np.dot(x, theta)

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 誤差の差分
diff = 1

# 学習を繰り返す
error = E(X, train_y)
while diff > 1e-2:
    # パラメータを更新
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 前回の誤差との差分を計算
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# 
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()