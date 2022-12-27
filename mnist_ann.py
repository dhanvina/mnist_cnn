# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
mnist_train = pd.read_csv("train.csv")
mnist_test = pd.read_csv("test.csv")

# %%
train_data_digit1 = np.asarray(mnist_train.iloc[0:1,1:]).reshape(28,28)
test_data_digit1 = np.asarray(mnist_test.iloc[0:1,]).reshape(28,28)

# %%
plt.subplot(1,2,1)
plt.imshow(train_data_digit1,cmap = plt.cm.gray_r)
plt.title("digit 1")


# %%
plt.imshow(test_data_digit1,cmap = plt.cm.gray_r)
plt.title("digit 2")

# %%
x_train = mnist_train.iloc[:,1:]
y_train = mnist_train.iloc[:,0:1]

# %%
from sklearn.neural_network import MLPClassifier

# %%
nn_model = MLPClassifier(hidden_layer_sizes=(50))
nn_model.fit(x_train,mnist_train.iloc[:,0])
print(nn_model.predict(mnist_test.iloc[0:1,]))

# %%
from sklearn.metrics import classification_report
print(classification_report(y_train,predict_digits))


