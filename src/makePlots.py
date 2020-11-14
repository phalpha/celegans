import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

training_loss = [0.0569, 0.0100, 0.0056, 0.0038, 0.0029, 0.0022, 0.0016, 0.0013, 0.0107, 0.0037]
validation_loss = [0.0634, 0.0595, 0.0486, 0.0612, 0.0670, 0.1110, 0.1405, 0.1574, 0.0309, 0.1219]
training_accuracy = [0.9847, 0.9963, 0.9979, 0.9986, 0.9990, 0.9993, 0.9995, 0.9996, 0.9985, 0.9986]
validation_accuracy = [0.9874, 0.9880, 0.9885, 0.9880, 0.9883, 0.9882, 0.9881, 0.9880, 0.9893, 0.9885]

epochs = [ i for i in range(10)]


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

confusion_matrix = [[110, 7],[12,145]]
cm = np.array(confusion_matrix)

plot_confusion_matrix(cm,['Alive','Dead'], title='Confusion matrix for Cell Statuss', cmap=None, normalize=True)


confusion_matrix_2 = [[8, 0],[0,12]]
cm = np.array(confusion_matrix_2)

plot_confusion_matrix(cm,['Untreated','Treated'], title='Confusion matrix for Treated vs. Untreated Images', cmap=None, normalize=True)


plt.plot(epochs, training_loss, label = "Training Loss")
plt.plot(epochs, validation_loss, label = "Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Loss over Epochs')
plt.legend()
plt.show()


plt.plot(epochs, training_accuracy, label = "Training Accuracy")
plt.plot(epochs, validation_accuracy, label = "Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('accuracy Value')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()
