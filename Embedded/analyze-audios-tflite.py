# D. Velasco-Montero, 2025

import sys
import os
import numpy as np
import librosa
import soundfile
import json
import tensorflow as tf
import logging as log
import datetime
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score, top_k_accuracy_score
#import seaborn as sns
from PIL import Image
from tqdm import tqdm
#from tensorflow.keras.models import load_model
from collections import Counter
import multiprocessing
import argparse

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)
import warnings
warnings.simplefilter("error", RuntimeWarning)

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../util'))
sys.path.append(PROJECT_ROOT)

import birdnet_util.audio as audio
from birdnet_util.audio0 import spectrogram  # Importación de la función de espectrograma

from  util import bcolors
   
# ---------------------- LOAD TRAINED MODEL ---------------------- #
def model_loading(MODEL_PATH, TFLITE_THREADS=1):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=TFLITE_THREADS)
    interpreter.allocate_tensors()
    print("[INFO] Model loaded successfully.")
    return interpreter

def load_labels(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
       LABELS = [line.strip() for line in f]

    print(f"# Target categories: {len(LABELS)}")
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

  
def analyze_folder(INPUT_PATH, OUTPUT_PATH, interpreter, LABELS, MIN_CONF, filename='predictions.txt', MAX_SEGMENTS=1000):

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
        
    listfiles = sorted(os.listdir(INPUT_PATH))
    print(bcolors.OKCYAN+ f"Found {len(listfiles)} files in {INPUT_PATH}"+bcolors.ENDC)
 
    for f in tqdm(listfiles):
        print(f"Analyzing {f}", flush=True)
        start_time = datetime.datetime.now()
        full_path = os.path.join(INPUT_PATH, f)
        chunk_preds = []
        print_preds = {}
        
        output_filename = os.path.splitext(f)[0] + "-predictions.txt"
        output_path = os.path.join(OUTPUT_PATH, output_filename)

        try:
            sig, rate = audio.openAudioFile(full_path, SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
            chunks = audio.splitSignal(sig, rate, SIG_LENGTH, SIG_OVERLAP, SIG_MINLEN)

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
                interpreter.set_tensor(input_details[0]['index'], img.astype(input_details[0]['dtype']))
                interpreter.invoke()

                # Obtener resultados
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predictions = np.squeeze(output_data)
                #predictions = model.predict(img, verbose=False)
                predicted_class = np.argmax(predictions)
                #print(predicted_class, LABELS[predicted_class], np.max(predictions))
          
                print_preds[f"{interval*SIG_LENGTH}-{(interval+1)*SIG_LENGTH}"] = predictions
                      
            # Mostrar las predicciones filtradas
            filtered_predictions = apply_confidence_threshold(print_preds, MIN_CONF)
            #print(filtered_predictions)
            with open(output_path, 'w') as out_f:
                for k, p_c in filtered_predictions.items():
                    predicted_class, confidence = p_c
                    label = LABELS[predicted_class] if predicted_class < len(LABELS) else f"class_{pred_class}"
                    out_f.write(f"{k.split('-')[0]}\t{k.split('-')[1]}\t{LABELS[predicted_class]}\t{confidence:.2f}\t{f}\n")
                
        except Exception as e:
            print(f"[Error] Cannot process audio file {os.path.join(INPUT_PATH, f)}: {e}")
            
        delta_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f"Finished {f} in {delta_time:.2f} seconds", flush=True)
                
       
    print(bcolors.OKCYAN+ f"Saved results into {OUTPUT_PATH}"+bcolors.ENDC)

    

    
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
MODEL_PATH = "../Models/mobilenet-224-337wi-ft.tflite" 
LABEL_FILE = "../Models/species-list-337.txt"
TFLITE_THREADS = max(1, multiprocessing.cpu_count() // 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", help="Input data", action="store", default='AUDIOSTFM/test_files/Accipiter gentilis/')
    parser.add_argument("--o", help="Output folder", action="store", default='OUTPUT_FOLDER')
    parser.add_argument("--min_conf", help="confidence threshold", action="store", default=0.5, type=float)
    args = parser.parse_args()
    
    INPUT_PATH = args.i
    OUTPUT_PATH = args.o
    MIN_CONF = args.min_conf

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)


    model = model_loading(MODEL_PATH, TFLITE_THREADS)
    LABELS = load_labels(LABEL_FILE)
    print(LABELS)
    
    analyze_folder(INPUT_PATH, OUTPUT_PATH, model, LABELS, MIN_CONF)


