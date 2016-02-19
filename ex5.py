import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model, preprocessing

data = scio.loadmat('ex5data1.mat')
X = data['X']
Xval = data['Xval']
Xtest = data['Xtest']
y = data['y']
yval = data['yval']
ytest = data['ytest']

model = linear_model.Ridge(alpha=0.0)
model.fit(X, y)

#px = np.array(np.linspace(np.min(X),np.max(X),100)).reshape(-1,1)
## minからmaxまでの範囲を100分割して、縦ベクトル（行列？）へ変換
#py = model.predict(px)
#plt.plot(px, py)
#plt.scatter(X,y)
#plt.show()

#--- 線形回帰でLearning Curveを描いてみる ---#
error_train = np.zeros(11)
error_val = np.zeros(11)
model = linear_model.Ridge(alpha=0.0)
for i in range(1,12):
    # 訓練データのサブセットi個のみで回帰を実施
    model.fit(X[0:i], y[0:i])
    # その訓練データのサブセットi個でのエラーを計算
    error_train[i-1] = sum((y[0:i] - model.predict(X[0:i]))**2) / (2*i)
    # 交差検定用データでのエラーのを計算
    error_val[i-1] = sum( (yval - model.predict(Xval) )**2 ) / (2*yval.size)

px = np.arange(1,12)
plt.plot(px, error_train, label='Train')
plt.plot(px, error_val, label='Cross Validation')
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()
