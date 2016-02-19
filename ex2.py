import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("ex2data1.txt", header = None)
X = np.array([data[0],data[1]]).T
y = np.array(data[2])

# numpy bool index
pos = (y==1)
neg = (y==0)

plt.scatter(X[pos,0], X[pos,1], marker='+', c='black')
plt.scatter(X[neg,0], X[neg,1], marker='o', c='y')
plt.legend(['Admitted', 'Not admitted'], scatterpoints=1)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")

model = linear_model.LogisticRegression(C=1000000.0)
model.fit(X,y)

[theta0] = model.intercept_
[[theta1, theta2]] = model.coef_

plot_x = np.array([min(X[:,0])-2, max(X[:,0])+2])
plot_y = -(theta0 + theta1*plot_x)/theta2
plt.plot(plot_x, plot_y, 'b')

plt.show()

## LogisticRegressionクラスでは、正則化の強さをパラメータCで指定する。
# 授業ではこれをパラメータλλで指定していたが、Cはλλの逆数という関係
# Cが小さいほど正則化を強く、大きいほど正則化を弱くすることができる。
# この例題では正則化なしにしたいため、Cには大きな値（1,000,000)を入れた。


## Decision Boundary（決定境界）を描画する。
# ロジスティック回帰の決定境界は θ^TX=0で定義される直線。
# 例題の場合、成分で書き下すと、theta0 + thea1*x1 + thtea2*x2 = 0
# x2 = -(theta0 + theta1*x1)/theta2　となる。

## python的ポイント
# y=1のものを+、y=0のものを○で散布図にプロットする際に、Numpyのブールインデックス
# （論理値によるインデックス）を使っている。

## Numpy Array
# 決定境界（直線）をプロットする際に、直線の両端の点を成分に持つ2次元のベクトルを
# plot_xに入れ、それに対応するplot_yをベクトル演算している。
# plot_xを作る際に、Numpy Arrayとして作成しないとベクトル演算ができないので注意。
