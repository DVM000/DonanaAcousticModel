import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import sys
import json
import matplotlib.pyplot as plt
import tqdm
import random
from augmentdata import data_augmentation
 
# ---------------------- PRINT ---------------------- #
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ---------------------- CHECK GPU AVAILABLE ---------------------- #
def check_GPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                 tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print(bcolors.WARNING+"WARNING: no GPU found."+bcolors.ENDC)
    print(gpus)
   
# ---------------------- DISTILLATION. MODEL CUSTOM FUNCTIONS ---------------------- #


# ---------------------- DISTILLATION. DATA LOADING ---------------------- #
#def load_data(image_path, npy_path, MIN_PER_CLASS, MAX_PER_CLASS, category_list=[], allow_missing_npy=True, ignore_npy = False):


# ---------------------- DATA PROCESSING ---------------------- #
IMG_HEIGHT = None
IMG_WIDTH = None
rescaling = None

def configure_image_preprocessing(height, width, scale, temp, a):
    global IMG_HEIGHT, IMG_WIDTH, rescaling, temperature, alpha
    IMG_HEIGHT = height
    IMG_WIDTH = width
    rescaling = scale
    temperature = temp
    alpha = a
    


# ---------------------- TRAIN MODEL ---------------------- #
def plot_history_1(acc,val_acc,loss,val_loss, namefig='fig1.png'):
  plt.figure(figsize=(12, 12))
  plt.subplot(2, 1, 1)
  N = len(acc) # total number of epochs
  plt.plot(np.arange(1,N+1), acc, '-o', label= "Training Accuracy")
  plt.plot(np.arange(1,N+1), val_acc, '-o', label= "Validation Accuracy")
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  #plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.text(1, acc[-1], "Training Accuracy: {:.2f}".format(acc[-1]))
  plt.text(1, val_acc[-1], "Val Accuracy: {:.2f}".format(val_acc[-1]))

  plt.subplot(2, 1, 2)
  plt.plot(np.arange(1,N+1), loss, '-s', label= "Training Loss")
  plt.plot(np.arange(1,N+1), val_loss, '-s', label= "Validation Loss")
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  #plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.text(1, loss[-1], "Training Loss: {:.2f}".format(loss[-1]))
  plt.text(1, val_loss[-1], "Val Loss: {:.2f}".format(val_loss[-1]))

  plt.show()
  plt.savefig(namefig, bbox_inches='tight')
  print(bcolors.OKCYAN+'Saved as ' + namefig +bcolors.ENDC)
  
# ---------------------- TEST MODEL ---------------------- #

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(true_classes, predIdxs, LABELS, FIGNAME='confusion_matrix.png'):
    print('Confusion Matrix')
    cm = confusion_matrix(true_classes, predIdxs) #, labels=LABELS)
    print(cm) 
    print('Accuracy {:.2f}%'.format( 100*sum( (predIdxs.squeeze()==true_classes))/ true_classes.shape[0] ) ) 
    cmP = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(60,60))
    cmP.plot(ax=ax, colorbar=False)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(cmP.im_,  cax=cax)
    plt.show()
    plt.savefig(FIGNAME)
    print(bcolors.OKCYAN+ f"Saved {FIGNAME}" +bcolors.ENDC)

#https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
import itertools
import re

def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu', FIGNAME='classification-report.png'):

    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []

    for line in lines[1:-4]:  # Excluir la última parte con los promedios
        match = re.match(r"(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if match:
            class_name, precision, recall, f1, sup = match.groups()
            classes.append(class_name.strip())
            class_names.append(class_name.strip())
            plotMat.append([float(precision), float(recall), float(f1)])
            support.append(int(sup))

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.figure(figsize=(10, 20))
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.tight_layout()
    plt.show()
    plt.savefig(FIGNAME)
    print(bcolors.OKCYAN+ f"Saved {FIGNAME}" +bcolors.ENDC)
 
