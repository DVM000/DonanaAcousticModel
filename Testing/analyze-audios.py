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

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../util'))
sys.path.append(PROJECT_ROOT)

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
    #model.summary()
    return model

def load_labels(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
       LABELS = [line.strip() for line in f]

    print(f"# Target categories: {len(LABELS)}")
    NUM_CLASSES = len(LABELS)
    return LABELS

def apply_confidence_threshold(predictions, threshold=0.5, top_only=False):
    """
    Filters predictions using a confidence threshold.

    - If top_only is True: keep only the highest prediction if it exceeds the threshold.
    - If threshold == -1: keep only the highest prediction (regardless of confidence).
    - Else: keep all predictions above the threshold.
    """
    filtered_predictions = {}

    for k, pred in predictions.items():
        pred = pred[0]  # remove batch dimension
        filtered_predictions[k] = []

        if threshold == -1:
            max_index = np.argmax(pred)
            max_conf = pred[max_index]
            filtered_predictions[k].append((max_index, max_conf))

        elif top_only:
            max_index = np.argmax(pred)
            max_conf = pred[max_index]
            if max_conf >= threshold:
                filtered_predictions[k].append((max_index, max_conf))

        else:
            for i, c in enumerate(pred):
                if c > threshold:
                    filtered_predictions[k].append((i, c))

    return filtered_predictions

  
def analyze_folder(INPUT_PATH, OUTPUT_PATH, model, LABELS, MIN_CONF, OVERLAP, top_only=False, filename='predictions.txt', MAX_SEGMENTS=1000):

    listfiles = sorted(os.listdir(INPUT_PATH))
    print(bcolors.OKCYAN+ f"Found {len(listfiles)} files in {INPUT_PATH}"+bcolors.ENDC)
 
    for f in tqdm(listfiles):
        full_path = os.path.join(INPUT_PATH, f)
        chunk_preds = []
        print_preds = {}
        
        output_filename = os.path.splitext(f)[0] + "-predictions.txt"
        output_path = os.path.join(OUTPUT_PATH, output_filename)
        start, end = 0, SIG_LENGTH

        try:
            sig, rate = audio.openAudioFile(full_path, SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
            chunks = audio.splitSignal(sig, rate, SIG_LENGTH, OVERLAP, SIG_MINLEN)

            for interval, y in enumerate(chunks[:MAX_SEGMENTS]):
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
                #print(predicted_class, LABELS[predicted_class], np.max(predictions))
          
                print_preds[f"{start}-{end}"] = predictions
                start += SIG_LENGTH - OVERLAP
                end = start + SIG_LENGTH 
                    
            # Mostrar las predicciones filtradas
            filtered_predictions = apply_confidence_threshold(print_preds, MIN_CONF, top_only)
            #print(filtered_predictions)
            with open(output_path, 'w') as out_f:
                for k, p_c in filtered_predictions.items():
                    for predicted_class, confidence in sorted(p_c, key=lambda x: -x[1]):
                        label = LABELS[predicted_class] if predicted_class < len(LABELS) else f"class_{pred_class}"
                        out_f.write(f"{k.split('-')[0]}\t{k.split('-')[1]}\t{LABELS[predicted_class]}\t{confidence:.2f}\t{f}\n")
                
        except Exception as e:
            print(f"[Error] Cannot process audio file {os.path.join(INPUT_PATH, f)}: {e}")
                
       
    print(bcolors.OKCYAN+ f"Saved results into {OUTPUT_PATH}"+bcolors.ENDC)

    

    
# ---------------------- Parameters ---------------------- #
SAMPLE_RATE = 48000
FILE_SPLITTING_DURATION = 600
BANDPASS_FMIN = 0
BANDPASS_FMAX = 15000
SIG_LENGTH = 3.0
#SIG_OVERLAP = 0
SIG_MINLEN = SIG_LENGTH #1.0
MAX_LIMIT = 1000
IMG_HEIGHT = 224
IMG_WIDTH = 224
rescaling = 1.0 / 255.0
MODEL_PATH = "../Models/mobilenet-224-337wi-ft.h5" 
LABEL_FILE = "../Models/species-list-337.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input data", action="store", default='AUDIOSTFM/test_files/Accipiter gentilis/')
    parser.add_argument("--o", help="Output folder", action="store", default='OUTPUT_FOLDER')
    parser.add_argument("--min_conf", help="confidence threshold", action="store", default=0.5, type=float)
    parser.add_argument("--overlap", help="Overlap of prediction segments", type=float, default=0.0)
    parser.add_argument("--top_only", help="If set, keep only top prediction per segment if it exceeds the threshold.", action="store_true")
    args = parser.parse_args()
    
    INPUT_PATH = args.i
    OUTPUT_PATH = args.o
    MIN_CONF = args.min_conf
    OVERLAP = args.overlap
    TOP_ONLY = args.top_only
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)


    model = model_loading(MODEL_PATH)
    LABELS = load_labels(LABEL_FILE)
    print(LABELS)
    
    analyze_folder(INPUT_PATH, OUTPUT_PATH, model, LABELS, MIN_CONF, OVERLAP, TOP_ONLY)


