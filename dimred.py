import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Generate a classification dataset and visualize it with scatter plot
X, y = datasets.make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, n_repeated=0, random_state=0)
# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1], c=y)

# Train-test split and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # Check the size of the training/testing dataset
# decision_tree = DecisionTreeClassifier(random_state=0)
# decision_tree.fit(X_train, y_train)
# print("Testing score (not shifted): {}".format(decision_tree.score(X_test, y_test))) # 0.884

# Trial on dataset shift
np.random.seed(0)
X_shift, y_shift = datasets.make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, n_repeated=0, shift=np.random.normal(size=(5,)), random_state=0)
# plt.subplot(122)
# plt.scatter(X_shift[:, 0], X_shift[:, 1], c=y_shift)

X_test_s1, X_test_s2, y_test_s1, y_test_s2 = train_test_split(X_shift, y_shift, test_size=0.8, random_state=1) # s1 is used for detection, s2 is for evaluation after correction
# print("Testing score (shifted): {}".format(decision_tree.score(X_test_s1, y_test_s1))) # 0.626, much lower than the non-shifted score

# plt.show() # Works the best for data with 2-dim only

# Trial on detecting a dataset shift: statistical-based approach
'''
# Trial
plt.subplot(131)
X_embedded = TSNE(random_state=0).fit_transform(X)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
plt.title("T-SNE")
plt.subplot(132)
X_pca = PCA(n_components=2).fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA")
plt.subplot(133)
X_kpca = KernelPCA(n_components=2, kernel="rbf").fit_transform(X)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title("Kernel PCA (RBF kernel)")
plt.show()
'''
plt.subplot(121)
X_detect = np.vstack((X_train, X_test_s1))
y_detect = np.hstack((np.ones(X_train.shape[0]), np.zeros(X_test_s1.shape[0])))
detect_embedded = TSNE(n_components=1, random_state=0).fit_transform(X_detect)
stat_detect, p_detect = ks_2samp(detect_embedded[y_detect == 0].squeeze(), detect_embedded[y_detect == 1].squeeze())
print("Detection has KS statistic = {} with p-value {}".format(stat_detect, p_detect)) # KS = 0.64
plt.scatter(detect_embedded, y_detect, c=y_detect)
plt.title("detection")
plt.subplot(122)
X_control = np.vstack((X_train, X_test))
y_control = np.hstack((np.ones(X_train.shape[0]), np.zeros(X_test.shape[0])))
control_embedded = TSNE(n_components=1, random_state=0).fit_transform(X_control)
stat_control, p_control = ks_2samp(control_embedded[y_control == 0].squeeze(), control_embedded[y_control == 1].squeeze())
print("Control has KS statistic = {} with p-value {}".format(stat_control, p_control)) # KS = 0.115
plt.scatter(control_embedded, y_detect, c=y_control)
plt.title("control")
plt.show()