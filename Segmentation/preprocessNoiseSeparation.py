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

import argparse


## functions
# --------------------------------------------------------------------------

# Si vamos a usar el espectrograma de BirdNet, mejor samplear a 48000 (con openAudioFile o indicandolo en librosa.load), 
#  o si no perderemos las altas frecuencias del espectrograma
    
from scipy.ndimage import binary_dilation, binary_erosion

#https://github.com/DeKUT-DSAIL/arm-dev-summit/blob/main/bioacoustics/baseline_models/audio_noise_separation.py

#These functions get the indices corresponding to audio and noise in a file

def compute_audio_mask(norm_specgram, sr, hop_len, category='audio'):
    """ Compute the section of signal corresponding to audio or noise
    This follows the approach described in
    Sprengel, E., Jaggi, M., Kilcher, Y., & Hofmann, T. (2016).
    Audio based bird species identification using deep learning techniques
    Args:
        norm_specgram: input spectrogram with values in range [0,1]
        hop_len: hop length used to generate the spectrogram
        category: whether 'audio' or 'noise'
    Returns:
        mask: the mask of samples belonging to 'category'
    Raises: ValueError if the category is not known
    """

    if category == 'audio':
        threshold = 3
    elif category == 'noise':
        threshold = 2.5
    else:
        raise ValueError('Unknown category')

    col_mask = norm_specgram > threshold * np.median(norm_specgram, axis=0)
    row_mask = norm_specgram.T > threshold * np.median(norm_specgram, axis=1)
    row_mask  = row_mask.T
    mask = col_mask & row_mask

    # erosion
    be_mask = binary_erosion(mask, np.ones((4, 4)))

    # dilation
    
    # pixeles/segundo en espectrograma = sr/ hop_length = 
    time_kernel_size = int(1 * sr / hop_len)
    
    bd_be_mask = binary_dilation(be_mask, np.ones((4, 4)))#time_kernel_size))) # 4))) # dilatamos el equivalente a 1 segundo 

    bd_be_mask = bd_be_mask.astype(int)
    selected_col = np.max(bd_be_mask, axis=0)
    bd_sel_col = binary_dilation(selected_col[:, None], np.ones((4, 1)))
    bd2_sel_col = binary_dilation(bd_sel_col, np.ones((4, 1)))

    
    # translate to audio samples
    selection_mtx = np.ones((norm_specgram.shape[1], hop_len)) * selected_col[:, None]

    audio_indx = selection_mtx.flatten().astype(bool)

    if category == 'audio':
        return audio_indx
    else:
        return ~audio_indx


def get_audio_noise(audio_array, nfft, hop_len):
    """ Get both the signal and noise
    Args:
        audio_array: an array of audio
        nfft: FFT length
        hop_len: hop length
    Returns:
        signal and noise
    """
    
    specgram = np.abs(librosa.stft(audio_array, n_fft=nfft, hop_length=hop_len))
    specgram_norm = specgram / (specgram.max() + 1e-8)
        
    audio_indx = compute_audio_mask(specgram_norm, SAMPLE_RATE, hop_len)[:len(audio_array)]
    #noise_indx = compute_audio_mask(specgram_norm, hop_len, 'noise')[:len(audio_array)]


    return audio_array[audio_indx]#, audio_array[noise_indx]
  
  
def split_and_spectrogram(PATH, PATHsave):
  
  listfiles = os.listdir(PATH)
  listfiles.sort()

  print('Found {} files in {}'.format(len(listfiles), PATH))
  
  for i,f in enumerate(tqdm(listfiles)): 
  
    if 1: #try:
        # Open file
        sig, rate = audio.openAudioFile(os.path.join(PATH,f), SAMPLE_RATE, offset=0, duration=FILE_SPLITTING_DURATION, fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
    
        y_sound = get_audio_noise(sig,512,128)
        
        # Split into raw audio chunks (3-sec intervals)
        chunks = audio.splitSignal(y_sound, rate, SIG_LENGTH, SIG_OVERLAP, SIG_MINLEN)

        for i,y in enumerate(chunks):
        
            # Normalize RMS
            '''rms =  np.sqrt(np.mean(np.abs(y)**2, axis=0, keepdims=True))
            try:
                y /= rms
            except RuntimeWarning:
                continue  '''  
            # Data augmentation (time domain) -> TO DO
           
            # Compute spectrogram
            spec,_ = spectrogram(y,rate)# shape=(128,128)) #,shape=(NMEL,RESH))#[..., np.newaxis]  # cambiado
            #spec = librosa.feature.melspectrogram(y=y, sr=rate, n_mels=224, hop_length=int(len(y) / (224 - 1)), fmin=BANDPASS_FMIN, fmax=BANDPASS_FMAX)
            #spec = librosa.power_to_db(spec)
            
            # Data augmentation (frequency domain) -> TO DO
        
            # Save image
            try:
                standardized_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec)) 
            except RuntimeWarning:
                continue
             #https://www.kaggle.com/code/frlemarchand/bird-song-classification-using-an-efficientnet/notebook
            spec_array = (np.asarray(standardized_spec.T)*255).astype(np.uint8)
            spec_image = Image.fromarray(spec_array.T)#.astype('uint8'))
            #spec_image =  Image.fromarray(np.array([spec_array, spec_array, spec_array]).T) # image
            spec_image.save("{}{}-{:03d}.png".format(PATHsave+'/',f.split('.')[0],i,len(chunks))) 
    else:#except:
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
	
	## Interval splitting + silence removal + spectrogram generation
	# --------------------------------------------------------------------------
	if proc=='all':
	 
	    split_and_spectrogram(PATH, PATHsave)
	  
