# D. Velasco-Montero, 2025

import sys
import os
import numpy as np
import librosa
import soundfile

import logging as log
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

import warnings
#warnings.simplefilter("error", RuntimeWarning)


import birdnet_util.audio as audio
from birdnet_util.audio0 import spectrogram
#import birdnet_util.config as cfg

import random

from PIL import Image
from tqdm import tqdm

import argparse

from maad import sound, rois, util

## functions
# --------------------------------------------------------------------------

# Si vamos a usar el espectrograma de BirdNet, mejor samplear a 48000 (con openAudioFile o indicandolo en librosa.load), 
#  en caso contrario perderemos las altas frecuencias del espectrograma
    
def agrupar_picos(valores, umbral=0.5):
    # Ordenar valores
    valores = sorted(valores)
    
    grupos = []
    grupo_actual = [valores[0]]

    for val in valores[1:]:
        if val - grupo_actual[-1] <= umbral:
            grupo_actual.append(val)
        else:
            grupos.append(np.mean(grupo_actual))  # Puedes cambiar a np.median(grupo_actual)
            grupo_actual = [val]

    grupos.append(np.mean(grupo_actual))  # Agregar el último grupo

    return grupos

 

def split_and_spectrogram(PATH, PATHsave):
	""" 
	    Extrae segmentos de audio de LEN segundos en torno a picos de energía 
	    y los guarda como archivos de audio.
	"""
	# Parámetros de la ventana de análisis
	frame_size = int(SAMPLE_RATE / 2)  # Ventana de 0.5s
	hop_length = frame_size // 2  # Desplazamiento de 50%
	    
	listfiles = os.listdir(PATH)
	listfiles.sort()

	print('Found {} files in {}'.format(len(listfiles), PATH))
	n_processed = 0
	
	for i,f in enumerate(tqdm(listfiles)): 
	
		if n_processed > MAX_LIMIT: 
		        print(f"Processed up to limmit {MAX_LIMIT}")   
		        break
    
		try:
			# Open file
			print(os.path.join(PATH,f))
			sig, rate = audio.openAudioFile(os.path.join(PATH,f), SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
			
			# Split into raw audio chunks (3-sec intervals)
			Sxx, tn ,fn ,ext = sound.spectrogram(sig, rate, nperseg=512, noverlap=256)
			Sxx_db = util.power2dB(Sxx, db_range=80)
			
			peak_time, peak_freq = rois.spectrogram_local_max(Sxx_db, tn, fn, ext, min_distance=int(0.1/(tn[1]-tn[0])), threshold_abs=TH_DB, display=False)  
			if not len(peak_time): print(f"NO Picos"); continue      
			peaks = agrupar_picos(peak_time, umbral=1.5)
			print(f"Encontrados {len(peaks)} picos")
			
			duration = librosa.get_duration(y=sig, sr=rate)
			
			# Extraer y guardar segmentos de audio en torno a los picos
			for i, peak_time in enumerate(peaks):
				start_time = max(0, peak_time - SIG_LENGTH / 2)  # Evitar valores negativos
				end_time = start_time + SIG_LENGTH #min(duration, peak_time + SIG_LENGTH / 2)
				
				if end_time > duration:
					end_time = duration
					start_time = end_time - SIG_LENGTH

				# Convertir tiempo a muestras
				start_sample = int(start_time * rate)
				end_sample = int(end_time * rate)

				# Extraer segmento
				y = sig[start_sample:end_sample]
				
			    	# Compute spectrogram
				spec,_ = spectrogram(y,rate, shape=(128,224)) #,shape=(NMEL,RESH))#[..., np.newaxis]  # cambiado
			
			    	# Normalize and save image
				try:
					standardized_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec)) 
				except RuntimeWarning:
					continue
			     	#https://www.kaggle.com/code/frlemarchand/bird-song-classification-using-an-efficientnet/notebook
				spec_array = (np.asarray(standardized_spec.T)*255).astype(np.uint8)
				spec_image = Image.fromarray(spec_array.T)#.astype('uint8'))
				spec_image.save("{}{}-{:03d}.png".format(PATHsave+'/',os.path.splitext(f)[0],i,len(peaks)))
				#spec_image.save("{}{}.png".format(PATHsave+'/',os.path.splitext(f)[0]))
				n_processed += 1
	    
		except:
			print(f"[Error] Cannot process audio file {os.path.join(PATH,f)}")      


## Parameters (see BirdNet config.py)
# --------------------------------------------------------------------------

SAMPLE_RATE = 48000		# input sample rate
FILE_SPLITTING_DURATION = 600 	# Number of seconds to load from a file at a time
BANDPASS_FMIN = 0 		# Settings for bandpass filter
BANDPASS_FMAX = 15000
SIG_LENGTH = 3.0		# input interval cambiado
SIG_OVERLAP = 0			# overlap between consecutive intervals
SIG_MINLEN = SIG_LENGTH 	# minimum length of audio chunk. For training, take SIG_LENGTH-duration chunks
TH_DB = -60                     # threshold to detect peaks. Lower, more detections     
MAX_LIMIT = 200                # maximum number of files to generate

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--i", help="Input data", action="store",
		            default = '/dataset/AUDIOS/TFM/train_files/Falco naumanni/')
	parser.add_argument("--o", help="Output folder", action="store",
		            default = './ejemplo/Falconaumanni/')
	parser.add_argument("--p", help="Data process", choices=['remove_silence','interval_split','spectrogram','birdnet_spectrogram', 'all'], action="store",
		            default = 'all')
		            
	args = parser.parse_args()
	PATH = args.i
	PATHsave = args.o
	proc = args.p

	if not os.path.exists(PATHsave):
            os.makedirs(PATHsave)
        
	print(PATHsave)
	
	## Interval splitting [+ silence removal] + spectrogram generation
	# --------------------------------------------------------------------------
	if proc=='all':
	    split_and_spectrogram(PATH, PATHsave)
	  
