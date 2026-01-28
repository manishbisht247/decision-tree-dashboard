import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def plot_confusion_matrix(ytest, ypred, title = 'Confusion Matrix'):

    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(ytest, ypred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax = ax)

    ax.set_title(title)
    return fig


def accuracy_vs_depth(X, y, max_depth_range = range(1, 21), testsize = 0.2):

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=testsize, random_state=42)

    train_accuracies = []
    test_accuracies = []

    for depth in max_depth_range:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(Xtrain, ytrain)

        ytrain_pred = clf.predict(Xtrain)
        ytest_pred = clf.predict(Xtest)

        train_accuracies.append(accuracy_score(ytrain, ytrain_pred))
        test_accuracies.append(accuracy_score(ytest, ytest_pred))

    fig, ax = plt.subplots()

    ax.plot(max_depth_range, train_accuracies, label='Training Accuracy', marker='o')
    ax.plot(max_depth_range, test_accuracies, label='Testing Accuracy', marker='o')

    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs. Decision Tree Max Depth')
    ax.legend()

    return fig














    
    







