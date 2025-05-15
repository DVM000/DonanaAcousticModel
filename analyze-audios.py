# D. Velasco-Montero, 2025

import sys
import os
import numpy as np
import librosa
import soundfile
import json
import tensorflow as tf
import logging as log
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model
from collections import Counter
import argparse

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
import warnings
warnings.simplefilter("error", RuntimeWarning)

import birdnet_util.audio as audio
from birdnet_util.audio0 import spectrogram  # Importación de la función de espectrograma

from  util import bcolors, plot_confusion_matrix


# ---------------------- LOAD TRAINED MODEL ---------------------- #
def distillation_loss(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    hard_labels = y_true[:, -num_classes:]
    soft_labels = y_true[:, :num_classes] 

    soft_labels = tf.cast(soft_labels, tf.float32)
    hard_labels = tf.cast(hard_labels, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    soft_labels_softmax = tf.nn.softmax(soft_labels / temperature, axis=-1)
    y_pred_softmax = tf.nn.softmax(y_pred / temperature, axis=-1)

    kl_loss = tf.keras.losses.KLDivergence()(soft_labels_softmax, y_pred_softmax)
    hard_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(hard_labels, y_pred)

    total_loss = alpha * hard_loss + (1 - alpha) * kl_loss
    return total_loss


def custom_accuracy(y_true, y_pred):
    num_classes = tf.shape(y_pred)[-1]
    hard_labels = y_true[:, -num_classes:]
    true_classes = tf.argmax(hard_labels, axis=-1)
    pred_classes = tf.argmax(y_pred, axis=-1)
    accuracy = tf.cast(tf.equal(true_classes, pred_classes), dtype=tf.float32)
    return tf.keras.backend.mean(accuracy)


def model_loading(MODEL_PATH):
    model = load_model(MODEL_PATH, custom_objects={"distillation_loss": distillation_loss, "custom_accuracy": custom_accuracy})
    model.summary()
    return model


def load_labels():
    with open(f'birdnet_idx.json', 'r') as fp:
        idx_dict = json.load(fp)

    with open("selected-species-model-all305.txt", "r") as f:
       LABELS = [line.strip() for line in f]

    print(f"Target categories {LABELS}")
    NUM_CLASSES = len(LABELS)
    return LABELS

def apply_confidence_threshold(predictions, threshold=0.5):
    """
    Filtra las predicciones usando un umbral de confianza.
    
    :param predictions: Las predicciones del modelo (probabilidades de cada clase).
    :param threshold: El umbral de confianza. Solo las clases con una probabilidad superior a este valor se mantienen.
    :return: Las clases predichas que superen el umbral y sus probabilidades.
    """
    filtered_predictions = {}
    for k,pred in predictions.items():
        max_prob = np.max(pred) # Salida de la red con mayor probabilidad
        if max_prob >= threshold:
            predicted_class = np.argmax(pred)  # Clase con mayor probabilidad
            filtered_predictions[k]=(predicted_class, max_prob)
    
    return filtered_predictions

from sklearn.metrics import accuracy_score
def top_k_accuracy(y_true, y_probs, k=5):
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)

def file_level_metrics(file_preds, file_true):
    TP = 0
    FP = 0
    FN = 0

    for pred, truth in zip(file_preds, file_true):
        if truth & pred:
            TP += 1
        else:
            FN += 1
        FP += len(pred - truth)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return precision, recall, TP, FP, FN  
    
def analyze_all(ROOT_PATH, model, LABELS, filename='predictions.txt'):

    fopen = open(filename,'w')
    
    y_true = []
    y_pred = []
    y_probs = []  # Guarda las probabilidades completas
    file_level_preds = []
    file_level_true = []

    for class_dir in sorted(os.listdir(ROOT_PATH))[:-1][:5]:
        class_path = os.path.join(ROOT_PATH, class_dir)
        if not os.path.isdir(class_path):
            continue  # Saltar archivos 

        try:
            true_class_idx = LABELS.index(class_dir)
        except ValueError:
            print(bcolors.WARNING+ f"Class {class_dir} not in LABELS. Skipping." +bcolors.ENDC)
            continue
              
        listfiles = sorted(os.listdir(class_path))
        print(bcolors.OKCYAN+ f"Found {len(listfiles)} files in {class_dir}"+bcolors.ENDC)

        print(f"\n{class_dir}:", file=fopen)
        
        for f in tqdm(listfiles[:5]):
            full_path = os.path.join(class_path, f)
            chunk_preds = []
            chunk_probs = []
            print_preds = {}

            if 1:#try:
                sig, rate = audio.openAudioFile(full_path, SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
                chunks = audio.splitSignal(sig, rate, SIG_LENGTH, SIG_OVERLAP, SIG_MINLEN)

                for interval, y in enumerate(chunks):
                    spec, _ = spectrogram(y, rate, shape=(128, 224))
                    try:
                        standardized_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec)) 
                    except RuntimeWarning:
                        continue

                    spec_array = (np.asarray(standardized_spec.T) * 255)#.astype(np.uint8)
                    img = Image.fromarray(spec_array.T)

                    # Preprocessing
                    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
                    img = np.expand_dims(img, axis=-1)  # Añadir dimensión de canal (1)
                    img = np.repeat(img, 3, axis=-1)  # Convertir a 3 canales
                    img = np.expand_dims(img, axis=0)  # Añadir batch dimension
                    img = img.astype(np.float32) * rescaling

                    # Model inference
                    predictions = model.predict(img, verbose=False)
                    predicted_class = np.argmax(predictions)
             
                    # Obtener etiqueta real a partir del nombre del archivo o directorio
                    #true_class = LABELS.index(PATH.split("/")[-2])  # Ajustar según convención de nombres
                    #y_true.append(0) #true_class)
                
                    #y_pred.append(predicted_class)
                    print_preds[f"{interval*SIG_LENGTH}-{(interval+1)*SIG_LENGTH}"] = predictions
                    
                    if np.max(predictions)>MIN_CONF:
                        chunk_preds.append(predicted_class)
                        chunk_probs.append(predictions[0])  # shape: (num_classes,)
                
                # Filtrar predicciones con el umbral de confianza
                filtered_predictions = apply_confidence_threshold(print_preds, MIN_CONF)

                # Mostrar las predicciones filtradas
                if filtered_predictions:
                    print(f"Predictions for {f}:", file=fopen)
                    for k, p_c in filtered_predictions.items():
                        predicted_class, confidence = p_c 
                        try:     print(f"{k.split('-')[0]}\t{k.split('-')[1]}\t{LABELS[predicted_class]}\t{confidence:.2f}\t{f}", file=fopen)
                        except:  print(f"{k.split('-')[0]}\t{k.split('-')[1]}\tclass {predicted_class}\t{confidence:.2f}\t{f}", file=fopen)
                else:
                    print(f"No prediction over {MIN_CONF} for {f}.", file=fopen)
                
                # Determinar clase mayoritaria
                if chunk_preds:
                    most_common_pred = Counter(chunk_preds).most_common(1)[0][0]
                    y_pred.append(most_common_pred)
                    y_true.append(true_class_idx)
                    
                    # Media de probabilidades como representación final
                    mean_probs = np.mean(chunk_probs, axis=0)
                    y_probs.append(mean_probs)
                    
                    # Todas las predicciones a nivel de recording
                    file_level_preds.append(set(chunk_preds))
                    file_level_true.append(set([true_class_idx]))

            else:#except Exception as e:
                print(f"[Error] Cannot process audio file {os.path.join(PATH, f)}: {e}")
        
    print(bcolors.OKCYAN+ f"Saved results into {filename}"+bcolors.ENDC)
    return y_true, y_pred, y_probs, file_level_preds, file_level_true
    

    
# ---------------------- Parameters ---------------------- #
SAMPLE_RATE = 48000
FILE_SPLITTING_DURATION = 600
BANDPASS_FMIN = 0
BANDPASS_FMAX = 15000
SIG_LENGTH = 3.0
SIG_OVERLAP = 0
SIG_MINLEN = SIG_LENGTH
MAX_LIMIT = 1000
IMG_HEIGHT = 224
IMG_WIDTH = 224
rescaling = 1.0 / 255.0
MODEL_PATH = "mobilenet_spectrogram_distill-all305.h5" #"mobilenet_spectrogram_distill11.h5"
MODEL_PATH = "mobilenet_spectrogram-all305-d05.h5" #"mobilenet_spectrogram_distill11.h5"
MIN_CONF = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--i", help="Input data", action="store", default='/media/delia/HDD/dataset/AUDIOS/TFM/train_files/Falco naumanni/')
    #parser.add_argument("--i", help="Input data", action="store", default='/media/delia/External\ HD/TFM/birdnet-output_train/segments/')
    parser.add_argument("--i", help="Input data", action="store", default='/media/delia/HDD/dataset/AUDIOS/TFM/WABAD/DATA_WABAD/DATA/')
    parser.add_argument("--o", help="Output file", action="store", default='predictions.txt')
    #parser.add_argument("--min_conf", help="confidence threshold", action="store", default=0.5, type=float)
    #parser.add_argument("--gt", help="Ground truth", action="store", default="Falco naumanni")
    args = parser.parse_args()
    
    PATH = args.i
    PATHsave = args.o

    if not os.path.exists(PATHsave):
        os.makedirs(PATHsave)

    print(PATHsave)

    model = model_loading(MODEL_PATH)
    LABELS = load_labels()
    '''try:
        y_true = LABELS.index(args.gt)
    except: 
        y_true = -1
    
    #y_pred = analyze(PATH, model, LABELS, y_true)
    y_pred = analyze(PATH, model, LABELS, y_true)
    plot_confusion_matrix(y_true*np.ones(len(y_pred)), y_pred, LABELS)
    print(f"Accuracy: {np.sum(np.array(y_pred)==y_true)/len(y_pred):.2f}")'''

    
    y_true, y_pred, y_probs, file_preds, file_true = analyze_all(PATH, model, LABELS, filename=PATHsave)
    plot_confusion_matrix(y_true, y_pred, LABELS, FIGNAME='confusion-audios.png')

    acc = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_pred)
    print(f"Accuracy total: {acc:.2f}")
    
    # Accuracy metrics
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)

    top1_acc = accuracy_score(y_true_arr, np.argmax(y_probs_arr, axis=1))
    top5_acc = top_k_accuracy_score(y_true_arr, y_probs_arr, k=5, labels=np.arange(y_probs_arr.shape[1]))

    print(f"Top-1 Accuracy: {top1_acc:.2f}")
    print(f"Top-5 Accuracy: {top5_acc:.2f}")
    
    # Métricas a nivel de fichero
    precision, recall, TP, FP, FN = file_level_metrics(file_preds, file_true)

    print(f"\n[RECORDING-LEVEL] Precision: {precision:.2f} | Recall: {recall:.2f}")
    print(f"[RECORDING-LEVEL] TP: {TP}  FP: {FP}  FN: {FN}")



