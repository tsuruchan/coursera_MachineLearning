import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = np.array(Image.open('bird_small.png')) # imgは128x128x3 の3次元行列 
img_flat = img.reshape(128*128,3)/255 # 16384x3 の2次元行列に変換

model = KMeans(n_clusters=16).fit(img_flat)
# 16個のクラスタに分類する


# 各ピクセルがどのクラスタに分類されたかは model.labels_（0-15の数字）
# 各クラスタの重心の値はmodel.cluster_centers_（16x3行列） に格納されています。
# これを使って、各ピクセルの値をクラスタの重心で置き換えた行列は、
# model.cluster_centers_[model.labels_]として作成することができます.
# （16384 x 3 の2次元行列）
# これを、画像で表示できる形の 128 x 128 x 3 の3次元行列に戻すため、
# reshape(128, 128, 3)します。

img_comp = model.cluster_centers_[model.labels_].reshape(128,128,3)
plt.imshow(img_comp)
