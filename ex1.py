import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("ex1data1.txt", header=None)

plt.scatter(data[0], data[1], marker='x', c='r')
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")

X = np.array([data[0]]).T
y = np.array(data[1])

model = linear_model.LinearRegression()
model.fit(X, y)

px = np.arange(X.min(),X.max(),.01)[:,np.newaxis]
py = model.predict(px)

print(px)
print(py)

plt.plot(px, py, color="blue", linewidth=2)
plt.show()

print(model.predict(35000))
print(model.predict(70000))


## plt.scatterについて
# maker:マーカの種類
# c:色
# s:サイズ
# 参考URL「http://stat.biopapyrus.net/python/scatter.html」

## numpyの１次元ベクトルについて
# Pythonでは1次元の縦ベクトルと横ベクトルが区別されない。
# 明示的に縦ベクトルを作成するには、
# np.array([[1,2,3,4,5]]).T 
# np.array([1,2,3,4,5])[:,np.newaxis)]
# ＜実行例＞
#>>> a = [1,2,3,4,5]
#>>> b = np.array(a)
#>>> array([1, 2, 3, 4, 5])
#>>> c = np.array([a])
#>>> array([[1, 2, 3, 4, 5]])
#>>> d = np.array([a]).T
#>>> array([[1],
#       [2],
#       [3],
#       [4],
#       [5]])
#>>> e = np.array(a).T
#>>> array([1, 2, 3, 4, 5])

## 単変数線形回帰
# 線形回帰モデルは、scikit-learn のsklearn.linear_model.LinearRegression()
# クラスを使用する。
# まずインスタンスを作成し、model.fit(X,y)で学習を実施。
# 学習結果である切片と傾きはmodel.intercept_とmodel.coef_のように取り出せる。
# モデルを使って新しいXの値に対して予測をするにはmodel.predict(X)とする。

## px = np.arange(X.min(),X.max(),.01)[:,np.newaxis]について。
# np.arangeを使うと連続データが作成できる。
# np.arange（最初の値、最後の値、刻み値）
# [:,np.newaxis]　そして縦ベクトルへ変換。
