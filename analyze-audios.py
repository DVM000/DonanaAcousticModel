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
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import load_model
import argparse

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
import warnings
warnings.simplefilter("error", RuntimeWarning)

import birdnet_util.audio as audio
from birdnet_util.audio0 import spectrogram  # Importación de la función de espectrograma


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

    with open("selected-species-model.txt", "r") as f:
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
    
def analyze(PATH, model, LABELS, y_true):
    listfiles = sorted(os.listdir(PATH))
    print('Found {} files in {}'.format(len(listfiles), PATH))

    y_true = []
    y_pred = []

    n_processed = 0

    for i, f in enumerate(tqdm(listfiles[:5])): 
        if n_processed > MAX_LIMIT: 
            print(f"Processed up to limit {MAX_LIMIT}")   
            break
        
        pred_file = {}
        if 1:#try:
            sig, rate = audio.openAudioFile(os.path.join(PATH, f), SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
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
                
                y_pred.append(predicted_class)

                n_processed += 1
                
                pred_file[f"{interval*SIG_LENGTH}-{(interval+1)*SIG_LENGTH}"] = predictions
                
            # Filtrar predicciones con el umbral de confianza
            filtered_predictions = apply_confidence_threshold(pred_file, MIN_CONF)

            # Mostrar las predicciones filtradas
            if filtered_predictions:
                print(f"Predictions for {f}:")
                for k, p_c in filtered_predictions.items():
                    predicted_class, confidence = p_c 
                    try:     print(f"{k.split('-')[0]}\t{k.split('-')[1]}\t{LABELS[predicted_class]}\t{confidence:.2f}\t{f}")
                    except:  print(f"{k.split('-')[0]}\t{k.split('-')[1]}\tclass {predicted_class}\t{confidence:.2f}\t{f}")
            else:
                print(f"No prediction over {MIN_CONF} for {f}.")

        else:#except Exception as e:
            print(f"[Error] Cannot process audio file {os.path.join(PATH, f)}: {e}")

    return y_pred


def plot_confusion_matrix(y_true, y_pred, LABELS):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Verdadera')
    plt.title('Matriz de Confusión')
    plt.show()
    plt.savefig('confusion-audios.png')
    print(cm)
    print(y_true, y_pred)


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
MODEL_PATH = "mobilenet_spectrogram_distill.h5" #"mobilenet_spectrogram_distill11.h5"
MIN_CONF = 0.99

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--i", help="Input data", action="store", default='/media/delia/HDD/dataset/AUDIOS/TFM/train_files/Falco naumanni/')
    parser.add_argument("--i", help="Input data", action="store", default='/media/delia/External\ HD/TFM/birdnet-output_train/segments/Falco\ naumanni/')
    parser.add_argument("--o", help="Output folder", action="store", default='./ejemplo/')
    #parser.add_argument("--min_conf", help="confidence threshold", action="store", default=0.5, type=float)
    parser.add_argument("--gt", help="Ground truth", action="store", default="Falco naumanni")
    args = parser.parse_args()
    
    PATH = args.i
    PATHsave = args.o

    if not os.path.exists(PATHsave):
        os.makedirs(PATHsave)

    print(PATHsave)

    model = model_loading(MODEL_PATH)
    LABELS = load_labels()
    try:
        y_true = LABELS.index(args.gt)
    except: 
        y_true = -1
    
    y_pred = analyze(PATH, model, LABELS, y_true)
    
    plot_confusion_matrix(np.ones(len(y_pred))*y_true, y_pred, LABELS)
    print(f"Accuracy: {np.sum(np.array(y_pred)==y_true)/len(y_pred):.2f}")


