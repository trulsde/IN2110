import os
import itertools

from pathlib import Path

isWindows = os.name == "nt"
hasNativeDisplay = "DISPLAY" in os.environ
hasJupyterDisplay = "MPLBACKEND" in os.environ

if isWindows or hasNativeDisplay or hasJupyterDisplay:
    GUI = True
else:
    import matplotlib as mpl
    mpl.use('Agg')
    GUI = False

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

from nltk.tokenize import sent_tokenize, _treebank_word_tokenizer

def discrete_color_map(n):
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(n)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, n)

    return cmap

def plot_class(x, y, labels, label, cmap):
    x, y, labels = zip(*((x[i], y[i], labels[i]) for i in range(x.shape[0])
                         if labels[i] == label))

    plt.scatter(x, y, data=labels, cmap=cmap, label=label, alpha=0.5)

def plot_class_3D(x, y, z, labels, label, cmap, ax):
    x, y, z, labels = zip(*((x[i], y[i], z[i], labels[i]) for i in range(x.shape[0])
                         if labels[i] == label))

    ax.scatter(x, y, z, cmap=cmap, label=label, alpha=0.5)

def save_plot(prefix):
    n = 0

    for filename in Path(".").glob("{}-*.png".format(prefix)):
        n = max((n, 1 + int(filename.stem.split("-")[-1])))

    out_file = '{}-{}.png'.format(prefix, n)

    print("Saving plot as '{}'.".format(out_file))

    plt.savefig(out_file)


def scatter_plot(x, labels):
    svd = TruncatedSVD(2).fit_transform(x)
    x, y = svd[:,0], svd[:,1]

    colors = {i: label for i,label in enumerate(set(labels))}

    cmap = discrete_color_map(len(colors))

    plt.figure()

    for i in range(len(colors)):
        plot_class(x, y, labels, colors[i], cmap)

    plt.legend()

    if GUI:
        plt.show()
    else:
        save_plot("scatter")

def scatter_plot_3D(x, labels):
    svd = TruncatedSVD(3).fit_transform(x)
    x, y, z = svd[:,0], svd[:,1], svd[:,2]

    colors = {i: label for i,label in enumerate(set(labels))}

    cmap = discrete_color_map(len(colors))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(colors)):
        plot_class_3D(x, y, z, labels, colors[i], cmap, ax)

    plt.legend()

    if GUI:
        plt.show()
    else:
        save_plot("scatter")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if GUI:
        plt.show()
    else:
        save_plot("confusion")
