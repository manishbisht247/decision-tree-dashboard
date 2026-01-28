from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_matrics(ytrain, ytrain_pred, ytest, ytest_pred):

    metrics = {
        # accuracy -> 1. Train and 2. Test
        'train_accuracy' : accuracy_score(ytrain, ytrain_pred),
        'test_accuracy' : accuracy_score(ytest, ytest_pred),

        # precision :
        'train_precision' : precision_score(ytrain, ytrain_pred, average = 'weighted'),
        'test_precision' : precision_score(ytest, ytest_pred, average = 'weighted'),
        
        # recall :
        'train_recall' : recall_score(ytrain, ytrain_pred),
        'test_recall' : recall_score(ytest, ytest_pred),

        # f1 score :
        'train_f1' : f1_score(ytrain, ytrain_pred),
        'test_f1' : f1_score(ytest, ytest_pred),

        # overfit_gap
        'overfit_gap' : abs(accuracy_score(ytrain, ytrain_pred) - accuracy_score(ytest, ytest_pred))
    }

    return metrics