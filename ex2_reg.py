import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing

data = pd.read_csv("ex2data2.txt", header = None)

x1 = np.array(data[0])
x2 = np.array(data[1])
y = np.array(data[2])

pos = (y==1)
neg = (y==0)
plt.scatter(x1[pos], x2[pos], marker='+', c='black')
plt.scatter(x1[neg], x2[neg], marker='o', c='y')
plt.legend(['y = 1','y = 0'], scatterpoints=1)
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")

# feature mapping
poly = preprocessing.PolynomialFeatures(6)
X = poly.fit_transform(np.c_[x1, x2])

# Logistic Regressin with L2_Reg
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y)

px = np.arange(-1.0, 1.5, 0.1)
py = np.arange(-1.0, 1.5, 0.1)
PX, PY = np.meshgrid(px, py)
XX = poly.fit_transform(np.c_[PX.ravel(), PY.ravel()])
Z = model.predict_proba(XX)[:,1] 
Z = Z.reshape(PX.shape)
plt.contour(PX, PY, Z, levels=[0.5], linewidths=3)
plt.show()


## feature maping
# feature mappingを使うとよりよい分類器へとなるが、overfittingしやすくなる。
# ゆえに、正規化をすることによってoverfitting対策とする。

## Reg
# L2正則化（リッジ回帰）なので、'penalty='l2'のオプションを入れる。
# 正則化の強さを変えて学習をしたモデルで決定境界を描いてみるが、
# Courseraではλ=0,1,100λ=0,1,100の3種類だったところを、
# Pythonの例ではC=1000000.0,C=1.0,C=0.01として描く。

## Decision Boundary
# PX,PYはそれぞれ 25x25 行列
# 特徴量マッピング 引数はravel()で625次元ベクトルに変換して渡す XXは 625x28行列
# ravel()メソッドで行列をベクトルに変換する。
# eshgridで作成した座標の行列は、mapFeatureやLogisticRegression.predict_proba()
# 渡す際にはベクトルにする必要があるた
# 逆に、返り値のベクトルZはreshape()メソッドで行列にもどす。
# ロジスティック回帰モデルで予測 y=1の確率は結果の2列目に入っているので取り出す
# Zは625次元ベクトル →　Zを25x25行列に変換
# Z=0.5の等高線が決定境界となる
