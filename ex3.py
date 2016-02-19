import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model

data = scio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].ravel()

model = linear_model.LogisticRegression(penalty='l2', C=10.0)
model.fit(X, y)

print(model.score(X, y))

# print negative example
wrong_index = np.array(np.nonzero(np.array([model.predict(X) != y]).ravel())).ravel()
wrong_sample_index = np.random.randint(0,len(wrong_index),25)
fig = plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
for i in range(0,25):
    ax = fig.add_subplot(5,5,i+1)
    ax.axis('off')
    ax.imshow(X[wrong_index[wrong_sample_index[i]]].reshape(20,20).T, cmap = plt.get_cmap('gray'))
    ax.set_title(str(model.predict(X[wrong_index[wrong_sample_index[i]]])[0]))
plt.show()

# 1行目：np.array([model.predict(X) != y])で、認識を間違えたらTrue、正解したらFalseが入った行列(5000x1）を取り出します。それをnp.nonzero()関数に入れると、Trueのデータのインデックスが入った行列が手に入ります。最終的にはベクトルで取り出したいので、.ravel()を2回使っています。
# 2行目：wrong_sample_indexは、1行目で取り出した間違えたインデックスの入ったベクトル（この場合は175個）の中から、ランダムで25個取り出すために作成したインデックスです。
#表示はpyplotのsubplotを使って、5x5のサブプロットで表示します。
#set_titleのところで、画像のタイトルとしてモデルがつけた（間違った）ラベルを表示します。ラベルはmodel.predict(X[wrong_index[wrong_sample_index[1]]])で取得しますが、1x1行列として返ってくるため、[0]をつけてスカラーとして取り出します。
