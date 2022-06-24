from ID3 import ID3
from utils import *

"""
Make the imports of python packages needed
"""

"""
========================================================================
========================================================================
                              Experiments 
========================================================================
========================================================================
"""
target_attribute = 'diagnosis'


def find_best_pruning_m(train_dataset: np.array, m_choices, num_folds=5):
    """
    Use cross validation to find the best M for the id3 model.

    :param train_dataset: Training dataset.
    :param m_choices: A sequence of possible value of M for the ID3 model min_for_pruning attribute.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_M, accuracies) where:
        best_M: the value of M with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each M (list of lists).
    """
    accuracies = []
    for i, m in enumerate(m_choices):
        model = ID3(label_names=attributes_names, min_for_pruning=m)
        m_acc = []
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=ID)
        for train, test in create_train_validation_split(train_dataset, kf):
                X_train, Y_train, X_test, Y_test = get_dataset_split(train, test, target_attribute)
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                m_acc.append(accuracy(Y_test, Y_pred))

        accuracies.append(np.mean(m_acc))

    best_m_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_m = m_choices[best_m_idx]

    return best_m, accuracies


def basic_experiment(x_train, y_train, x_test, y_test, formatted_print=False):
    """
    Use ID3 model, to train on the training dataset and evaluating the accuracy in the test set.
    """
    tree = ID3(attributes_names)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    acc = accuracy(y_test, y_pred)

    assert acc > 0.9, 'you should get an accuracy of at least 90% for the full ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)


def cross_validation_experiment(plot_graph=True):
    """
    Use cross validation to find the best M for the ID3 model, used as pruning parameter.

    :param plot_graph: either to plot or not the experiment result, default is True
    :return: best_m: the value of M with the highest mean accuracy across folds
    """
    best_m = None
    accuracies = []
    m_choices = [50, 70, 90, 110, 130, 150, 170, 190]
    num_folds = 5
    best_m, accuracies = find_best_pruning_m(train_dataset, m_choices, num_folds)

    accuracies_mean = np.array([np.mean(acc) * 100 for acc in accuracies])
    if plot_graph:
        util_plot_graph(x=m_choices, y=accuracies_mean, x_label='M', y_label='Validation Accuracy %')
        print('{:^10s} | {:^10s}'.format('M value', 'nValidation Accuracy'))
        for i, m in enumerate(m_choices):
            print('{:^10d} | {:.2f}%'.format(m, accuracies_mean[i]))
        print(f'===========================')
        # Calculate accuracy
        accuracy_best_m = accuracies_mean[m_choices.index(best_m)]
        print('{:^10s} | {:^10s}'.format('Best M', 'Validation Accuracy'))
        print('{:^10d} | {:.2f}%'.format(best_m, accuracy_best_m))

    return best_m


# ========================================================================
def best_m_test(x_train, y_train, x_test, y_test, min_for_pruning):
    """
        Test the pruning for the best M value we have got from the cross validation experiment.
        :param: best_m: the value of M with the highest mean accuracy across folds
        :return: acc: the accuracy value of ID3 decision tree instance that using the best_m as the pruning parameter.
    """
    tree = ID3(attributes_names, min_for_pruning=min_for_pruning)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    return accuracy(y_test, y_pred)


# ========================================================================
if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    data_split = get_dataset_split(train_dataset, test_dataset, target_attribute)

    """
    Usages helper:
    (*) To get the results in “informal” or nicely printable string representation of an object
        modify the call "utils.set_formatted_values(value=False)" from False to True and run it
    """
    formatted_print = True
    basic_experiment(*data_split, formatted_print)

    """
       cross validation experiment
       (*) To run the cross validation experiment over the  M pruning hyper-parameter 
           uncomment below code and run it
           modify the value from False to True to plot the experiment result
    """
    plot_graphs = True
    best_m = cross_validation_experiment(plot_graph=plot_graphs)
    print(f'best_m = {best_m}')

    """
        pruning experiment, run with the best parameter
        (*) To run the experiment uncomment below code and run it
    """
    acc = best_m_test(*data_split, min_for_pruning=best_m)
    assert acc > 0.95, 'you should get an accuracy of at least 95% for the pruned ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)
