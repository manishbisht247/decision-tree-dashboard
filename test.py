import data.data_loader as dd
from sklearn.model_selection import train_test_split
from model.decision_tree import train_decision_tree
from utils.metrics import calculate_matrics
from utils.plots import plot_confusion_matrix, accuracy_vs_depth
import matplotlib.pyplot as plt

X, y, _, _ = dd.data_loader()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_decision_tree(Xtrain, ytrain)

ytrain_pred = model.predict(Xtrain)
ytest_pred = model.predict(Xtest)

fig1 = plot_confusion_matrix(ytest, ytest_pred)
fig2 = accuracy_vs_depth(X, y)

plt.show()