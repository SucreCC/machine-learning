import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from k_means import KMeans

data = pd.read_csv('./data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

x_axis = 'petal_length'
y_axis = 'petal_width'

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data[x_axis][:], data[y_axis][:])
plt.title('label unknown')
plt.show()
