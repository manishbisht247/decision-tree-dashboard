from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(Xtrain, ytrain, max_depth = None, min_samples_split = 3, min_samples_leaf = 2, criterion = 'gini'):


    model = DecisionTreeClassifier(
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        criterion = criterion,
        random_state = 42
    )

    model.fit(Xtrain, ytrain)

    return model