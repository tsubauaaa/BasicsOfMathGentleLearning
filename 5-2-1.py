import numpy as np
import matplotlib.pyplot as plt

# 学習データを行列として読み込む
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)

# 1列目の抽出
train_x = train[:,0]
# 2列目の抽出
train_y = train[:,1]

# プロット
plt.plot(train_x, train_y, 'o')
plt.show()