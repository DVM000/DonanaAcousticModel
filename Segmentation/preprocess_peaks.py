# D. Velasco-Montero, 2025

import sys
import os
import numpy as np
import librosa
import soundfile

import logging as log
log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

import warnings
warnings.simplefilter("error", RuntimeWarning)


import birdnet_util.audio as audio
from birdnet_util.audio0 import spectrogram
#import birdnet_util.config as cfg


from PIL import Image
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
        
import argparse



def rms(x):
    """Computes signal RMS of x"""
    return np.sqrt(np.mean(x**2))

def dBFS(x):
    """calculate the root-mean-square dB value (relative to a full-scale sine wave)"""
    return 20 * np.log10(rms(x))
    
def rms_to_db(rms):
    return 20 * np.log10(rms)
     
def show(PATH):
  
	y, sr = librosa.load(PATH, sr=None)

	#  Calcular el nivel de RMS o dB medio para cada segundo de la señal
	duration = librosa.get_duration(y=y, sr=sr)
	frame_size = int(sr/2) # Ventana de 0.5 segundos
	overlap = frame_size // 2
	rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=overlap).flatten()
	rms_db = rms_to_db(rms)

	# Create the time axis for the RMS
	time_rms = np.linspace(0, duration, len(rms_db))

	# Plot the audio signal, intervals, and RMS
	f, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1.5, 1]})
	#plt.figure(figsize=(15, 8))

	# Plot the audio signal
	plt.subplot(2, 1, 1)
	plt.plot(np.linspace(0, len(y) / sr, len(y)), y, alpha=0.6)
	plt.title(f'Audio signal {PATH}')
	plt.xlabel('Time (s)')
	plt.ylabel('Amplitude')
	plt.xlim([0, duration])


	# Plot the RMS level or dB as points
	plt.subplot(2, 1, 2)
	plt.scatter(time_rms[1:-1], rms_db[1:-1], label='RMS Level (dB)', alpha=0.6) 
	#plt.title('RMS Level')
	plt.xlabel('Time (s)')
	plt.ylabel('RMS (dB)')
	plt.xlim([0, duration])
	#plt.legend()

        # peak detector
	#pidx = signal.find_peaks_cwt(rms_db, 1)
	pidx, _ = signal.find_peaks(rms_db, height=np.mean(rms_db)+np.std(rms_db))
	print(pidx)
	if len(pidx): 
		#plt.scatter(time_rms[pidx], [rms_db[int(k)] for k in time_rms[pidx]-1], color='red')
		plt.scatter(time_rms[pidx], rms_db[pidx], color='red')
	
	plt.tight_layout()
	plt.show()
	plt.savefig('signal_peaks.png')
	#plt.savefig('signal_dB_intervals_ebd.png')



def extract_audio_segments(PATH, LEN=3.0, save_path="segments"):
    """ 
    Extrae segmentos de audio de LEN segundos en torno a picos de energía 
    y los guarda como archivos de audio.
    """
    
    # Cargar audio
    y, sr = librosa.load(PATH, sr=None)
    
    # Parámetros de la ventana de análisis
    frame_size = int(sr / 2)  # Ventana de 0.5s
    hop_length = frame_size // 2  # Desplazamiento de 50%
    
    # Calcular energía RMS y convertir a dB
    rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length).flatten()
    rms_db = rms_to_db(rms)
    
    # Crear eje de tiempo para los RMS
    duration = librosa.get_duration(y=y, sr=sr)
    time_rms = np.linspace(0, duration, len(rms_db))
    
    # Detección de picos en la energía RMS
    peaks, _ = signal.find_peaks(rms_db, height=np.mean(rms_db) + np.std(rms_db))  # Picos por encima de la media + 1 desviación estándar
    
    # Convertir los picos a tiempos en segundos
    peak_times = time_rms[peaks]
    
    # Figura
    plt.figure(figsize=(15, 6))
    plt.plot(time_rms, rms_db, label="RMS (dB)", alpha=0.7)
    plt.scatter(peak_times, rms_db[peaks], color='red', label="Picos detectados")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("RMS (dB)")
    plt.title("Detección de picos de energía")
    plt.legend()
    plt.show()
    plt.savefig('signal_peaks2.png')

    # Extraer y guardar segmentos de audio en torno a los picos
    for i, peak_time in enumerate(peak_times):
        start_time = max(0, peak_time - LEN / 2)  # Evitar valores negativos
        end_time = min(duration, peak_time + LEN / 2)

        # Convertir tiempo a muestras
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extraer segmento
        segment = y[start_sample:end_sample]

        # Guardar el segmento de audio
        filename = f"{save_path}/segment_{i+1}.wav"
        sf.write(filename, segment, sr)
        print(f"Segmento guardado: {filename}")


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
	  
	for i,f in enumerate(tqdm(listfiles)): 
    
		try:
			# Open file
			sig, rate = audio.openAudioFile(os.path.join(PATH,f), SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
			
			# Calcular energía RMS y convertir a dB
			rms = librosa.feature.rms(y=sig, frame_length=frame_size, hop_length=hop_length).flatten()
			rms_db = rms_to_db(rms)
		    
		    	# Crear eje de tiempo para los RMS
			duration = librosa.get_duration(y=sig, sr=rate)
			time_rms = np.linspace(0, duration, len(rms_db))
		    
		    	# Detección de picos en la energía RMS
			peaks, _ = signal.find_peaks(rms_db, height=np.mean(rms_db) + np.std(rms_db))  # Picos por encima de la media + 1 desviación estándar
		    
		    	# Convertir los picos a tiempos en segundos
			peak_times = time_rms[peaks]
		   
		    	# Extraer y guardar segmentos de audio en torno a los picos
			for i, peak_time in enumerate(peak_times):
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
				
				# Normalize RMS
				'''rms =  np.sqrt(np.mean(np.abs(y)**2, axis=0, keepdims=True))
				try:
					y /= rms
				except RuntimeWarning:
					continue '''   
			
			    	# Compute spectrogram
				spec,_ = spectrogram(y,rate) #,shape=(128,128))
			
			    	# Save image
				try:
					standardized_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec)) 
				except RuntimeWarning:
					continue
			     	#https://www.kaggle.com/code/frlemarchand/bird-song-classification-using-an-efficientnet/notebook
				spec_array = (np.asarray(standardized_spec.T)*255).astype(np.uint8)
				spec_image = Image.fromarray(spec_array.T)#.astype('uint8'))
			    	#spec_image =  Image.fromarray(np.array([spec_array, spec_array, spec_array]).T) # image
				spec_image.save("{}{}-{:03d}.png".format(PATHsave+'/',f.split('.')[0],i,len(peaks))) # '.png'))
	    
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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--i", help="Input data", action="store",
		            default = '/dataset/AUDIOS/TFM/val_files/Falco naumanni/')#94925.mp3')
	parser.add_argument("--o", help="Output folder", action="store",
		            default = './ejemplo/Falconaumanni/')
		            	           
	args = parser.parse_args()
	PATH = args.i
	PATHsave = args.o
		
	#show(PATH)
	#extract_audio_segments(PATH)

	if not os.path.exists(PATHsave):
            os.makedirs(PATHsave)
        
	print(PATHsave)
	
	## Interval splitting [+ silence removal] + spectrogram generation
	# --------------------------------------------------------------------------
	split_and_spectrogram(PATH, PATHsave)
