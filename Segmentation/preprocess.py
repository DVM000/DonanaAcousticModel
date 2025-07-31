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

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../util'))
sys.path.append(PROJECT_ROOT)

import birdnet_util.audio as audio
from birdnet_util.audio0 import spectrogram #, spectrogram224
#import birdnet_util.config as cfg


from PIL import Image
from tqdm import tqdm

import argparse


## functions
# --------------------------------------------------------------------------

# Si vamos a usar el espectrograma de BirdNet, mejor samplear a 48000 (con openAudioFile o indicandolo en librosa.load), 
#  en caso contrario perderemos las altas frecuencias del espectrograma
    
def silence_removal(x,top_db=60, frame_length=2048, hop_length=512):
  import librosa
  split_sound = librosa.effects.split(x, top_db=top_db, frame_length=int(frame_length), hop_length=int(hop_length))         
  splits = []
  for idxs in split_sound:
    splits.append(x[idxs[0]:idxs[1]])
  return splits
  
def rms(x):
    """Computes signal RMS of x"""
    return np.sqrt(np.mean(x**2))

def dBFS(x):
    """calculate the root-mean-square dB value (relative to a full-scale sine wave)"""
    return 20 * np.log10(rms(x))
    
  
def split_and_spectrogram(PATH, PATHsave, sufix=False):
  
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
        sig, rate = audio.openAudioFile(os.path.join(PATH,f), SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
    
        # Split into raw audio chunks (3-sec intervals)
        chunks = audio.splitSignal(sig, rate, SIG_LENGTH, SIG_OVERLAP, SIG_MINLEN)
   
        for i,y in enumerate(chunks):
 
            # 2- Remove silence
            '''splits = silence_removal(y, top_db=57)#, frame_length=3*rate, hop_length=0.5*rate)'''
            '''db = dBFS(y)
            if db < -57: 
                continue'''
        
            # Normalize RMS
            '''rms =  np.sqrt(np.mean(np.abs(y)**2, axis=0, keepdims=True))
            try:
                y /= rms
            except RuntimeWarning:
                continue '''   
           
            # Compute spectrogram
            # more than 64 filters give us: UserWarning: Empty filters detected in mel frequency basis. Some channels will produce empty responses. Try increasing your sampling rate (and fmax) or reducing n_mels.
            spec,_ = spectrogram(y,rate, shape=(128,224)) #shape=(64,384)) # shape=(128,128)) 
            
            # Other options:
            #spec = librosa.feature.melspectrogram(y=y, sr=rate, n_mels=224, hop_length=int(len(y) / (224 - 1)), fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
            #spec = librosa.power_to_db(spec)
            #spec = librosa.feature.melspectrogram(y=y, sr=rate, n_mels=128)

            # Normalize and Save image
            try:
                standardized_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec)) 
            except RuntimeWarning:
                continue
             #https://www.kaggle.com/code/frlemarchand/bird-song-classification-using-an-efficientnet/notebook
            spec_array = (np.asarray(standardized_spec.T)*255).astype(np.uint8)
            spec_image = Image.fromarray(spec_array.T)#.astype('uint8'))
            #spec_image =  Image.fromarray(np.array([spec_array, spec_array, spec_array]).T) # 3-channel image
            if sufix:  spec_image.save("{}{}-{:03d}.png".format(PATHsave+'/',os.path.splitext(f)[0],i,len(chunks))) # '.png')) # Add sufix indicating number of segment
            else:      spec_image.save("{}{}.png".format(PATHsave+'/',os.path.splitext(f)[0])) # '.png')) # Do NOT add sufix indicating number of segment
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
MAX_LIMIT = 1000                # maximum number of files to generate

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--i", help="Input data", action="store",
		            default = '/dataset/AUDIOS/TFM/train_files/Falco naumanni/')
	parser.add_argument("--o", help="Output folder", action="store",
		            default = './ejemplo/Falconaumanni/')
	parser.add_argument("--p", help="Data process", choices=['remove_silence','interval_split','spectrogram','birdnet_spectrogram', 'all'], action="store",
		            default = 'all')
	parser.add_argument("--sufix", help="Add idx sufix to img name", action="store", type=bool,
		            default = False)
		            
	args = parser.parse_args()
	PATH = args.i
	PATHsave = args.o
	proc = args.p
	sufix = args.sufix

	if not os.path.exists(PATHsave):
            os.makedirs(PATHsave)
        
	print(PATHsave)
	
	## Interval splitting [+ silence removal] + spectrogram generation
	# --------------------------------------------------------------------------
	if proc=='all':
	    split_and_spectrogram(PATH, PATHsave, sufix)
	  
