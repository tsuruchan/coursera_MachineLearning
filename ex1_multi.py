import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("ex1data2.txt", header=None)
X = np.array([data[0],data[1]]).T
y = np.array(data[2])

model = linear_model.LinearRegression()
model.fit(X, y)

a = [1650,3]

print(model.predict(a))


# Theta computed from gradient descent:
# [ 334302.06399328  100087.11600585    3673.54845093]
# Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
#  $289314.620338
# Solving with normal equations...
# Theta computed from the normal equations:
# [ 89597.90954435    139.21067402  -8738.01911278]
# Predicted price of a 1650 sq-ft, 3 br house (using normal equations):
#  $293081.464335
# 以上より、正規方程式によって解を求めていると思われる。
# 一般には、n<10^4なら正規方程式
#          n>10^6なら最急降下法を使うと良いとされている。
