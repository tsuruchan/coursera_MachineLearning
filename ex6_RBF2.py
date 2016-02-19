import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm

# scipy.io.loadmat()を使ってmatlabデータを読み込み
data = scio.loadmat('ex6data3.mat')
X = data['X']
y = data['y'].ravel()
Xval = data['Xval']
yval = data['yval'].ravel()

c_values = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
gamma_values = 1/ (2* c_values**2)

scores = np.zeros([8,8])
for i_c in range(0,8):
    for i_g in range(0,8):
        model = svm.SVC(C=c_values[i_c], gamma=gamma_values[i_g], kernel='rbf')
        model.fit(X, y)
        scores[i_c, i_c] = model.score(Xval, yval)

# Scoreが最大のC,gammaを求める
# 多次元の場合配列から最大値のあるインデックスのタプルを返す
max_idx = np.unravel_index(np.argmax(scores), scores.shape)
model = svm.SVC(C=c_values[max_idx[0]], gamma=gamma_values[max_idx[1]], \
                kernel='rbf', probability=True)
model.fit(X, y)

# 交差検定データをプロットする
pos = (yval==1) # numpy bool index
neg = (yval==0) # numpy bool index
plt.scatter(Xval[pos,0], Xval[pos,1], marker='+', c='k')
plt.scatter(Xval[neg,0], Xval[neg,1], marker='o', c='y')

# Decision Boundary(決定境界)をプロットする
px = np.arange(-0.6, 0.25, 0.01)
py = np.arange(-0.8, 0.6, 0.01)
PX, PY = np.meshgrid(px, py) # PX,PYはそれぞれ 100x100 行列
XX = np.c_[PX.ravel(), PY.ravel()] # XXは 10000x2行列
Z = model.predict_proba(XX)[:,1] # SVMモデルで予測。y=1の確率は結果の2列目に入っているので取り出す。Zは10000次元ベクトル
Z = Z.reshape(PX.shape) # Zを100x100行列に変換
plt.contour(PX, PY, Z, levels=[0.5], linewidths=3) # Z=0.5の等高線が決定境界となる
plt.show()
