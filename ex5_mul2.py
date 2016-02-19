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

lambda_values = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
lambda_values2 = np.arange(3.5, 4.0, 0.001)

# Xの階乗を計算して新しい特徴量 X_poly とする
# Xは m x 1行列、X_polyは m x 8 行列
poly = preprocessing.PolynomialFeatures(degree=8, include_bias=False)
X_poly = poly.fit_transform(X)

# X_polyを使って線形回帰,最適なalphaを求める(RidgeCrossValidation)
model = linear_model.RidgeCV(alphas = lambda_values2)
model.fit(X_poly, y)

print(model.alpha_)
