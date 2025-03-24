# https://zenodo.org/records/7115878

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import os

# SHIFT AND STRETCH

# Frequency stretch
def y_stretch(input_sg, drop_rate = 0.10): # drop_rate: What fraction of the image is cut of from upper and lower ends at maximum
    orig_height = input_sg.shape[0]
    low = np.random.randint(0,drop_rate*orig_height)
    high = orig_height-np.random.randint(0,drop_rate*orig_height)
    input_sg = input_sg[low:high, :]
    x = np.arange(0, input_sg.shape[0])
    fit = scipy.interpolate.interp1d(x, input_sg, axis=0)
    input_sg = fit(np.linspace(0, input_sg.shape[0]-1, orig_height))
    return input_sg
    
# Time stretch
def x_stretch(input_sg, drop_rate = 0.10):  # drop_rate: What fraction of the image is cut of from left and right ends at maximum
    orig_length = input_sg.shape[1]
    left = np.random.randint(0,drop_rate*orig_length)
    right = orig_length-np.random.randint(0,drop_rate*orig_length)
    input_sg = input_sg[:, left:right]
    x = np.arange(0, input_sg.shape[1])
    fit = scipy.interpolate.interp1d(x, input_sg, axis=1)
    input_sg = fit(np.linspace(0, input_sg.shape[1]-1, orig_length))
    return input_sg

# Frequency shift
def y_shift(input_sg, shift_rate = 0.1):  # drop_rate: What fraction of the image is moved to higher/lower end of the image
    # move the spectrogram vertically and fill the remaining space with random noise
    orig_height = input_sg.shape[0]
    shift = np.random.randint(0,shift_rate*orig_height)
    new_file = np.empty(input_sg.shape)
    if(np.random.rand() < 0.5): # Are frequency bins moved up or down 
        new_file[0:orig_height-shift, :] = input_sg[shift:orig_height, :] 
        empty_aug = input_sg[orig_height-shift:orig_height, :]
        np.random.shuffle(empty_aug)
        new_file[orig_height-shift:orig_height, :] = empty_aug 
    else:
        new_file[shift:orig_height, :] = input_sg[0:orig_height-shift, :]
        empty_aug = input_sg[0:shift, :]
        np.random.shuffle(empty_aug)
        new_file[0:shift, :] = empty_aug
    return new_file
    
# Time shift
def x_shift(input_sg, shift_rate = 0.1):  # drop_rate: What fraction of the image is moved to another location in x-axis
    # move part of the spectrogram horizontally
    cut_point = np.random.randint(0, input_sg.shape[1])
    left = input_sg[:, 0:cut_point]
    right = input_sg[:, cut_point:input_sg.shape[1]]
    if(np.random.rand() < 0.5):  # is the left or right side of the spectrogram modified
        orig_length = right.shape[1]
        shift = np.random.randint(0,np.max([shift_rate*orig_length,1]))
        modified_right = np.empty(right.shape)
        if(np.random.rand() < 0.5): 
            modified_right[:, 0:orig_length-shift] = right[:, shift:orig_length] 
            modified_right[:, orig_length-shift:orig_length] = right[:, 0:shift]
        else:
            modified_right[:, 0:shift] = right[:, orig_length-shift:orig_length] 
            modified_right[:, shift:orig_length] = right[:, 0:orig_length-shift]
        right = modified_right
    else: 
        orig_length = left.shape[1]
        shift = np.random.randint(0, np.max([shift_rate*orig_length,1]))
        modified_left = np.empty(left.shape)
        if(np.random.rand() < 0.5): 
            modified_left[:, 0:orig_length-shift] = left[:, shift:orig_length] 
            modified_left[:, orig_length-shift:orig_length] = left[:, 0:shift]
        else:
            modified_left[:, 0:shift] = left[:, orig_length-shift:orig_length] 
            modified_left[:, shift:orig_length] = left[:, 0:orig_length-shift]
        left = modified_left
    return np.concatenate((left, right), axis=1)

# TIME AND FREQUENCY MASKS

# Frequency mask
def frequency_mask(input_sg, lamb = 0.5, max_height = 0.2): # lamb = lambda for Poisson distribution for the number of masks, max_height = maximum height for a single mask in proportion to the image height
    # sample the number of masks and heights for masks
    r = np.random.poisson(lamb)
    if(r > 0):
        for j in range(r):
            low,high = np.sort((j*input_sg.shape[0]/r)+np.random.randint(0, input_sg.shape[0]/r, 2))
            # Clip too high masks
            if(np.random.rand()<0.5):
                high = np.min([high, low + max_height*input_sg.shape[0]])
            else:
                low = np.max([low, high - max_height*input_sg.shape[0]])
            input_sg[int(low):int(high), :] = np.min(input_sg)
    return input_sg

# Time mask
def time_mask(input_sg, lamb = 0.5, max_length = 0.25): # lamb = lambda for Poisson distribution for the number of masks, max_lengtt = maximum length for a single mask in proportion to the image length
    # sample the number of masks and lengths for masks
    r = np.random.poisson(lamb)
    if(r > 0):
        for j in range(r):
            start,stop = np.sort((j*input_sg.shape[1]/r)+np.random.randint(0, input_sg.shape[1]/r, 2))
            # Clip too long masks
            if(np.random.rand()<0.5):
                stop = np.min([stop, start + max_length*input_sg.shape[1]])
            else:
                start = np.max([start, stop - max_length*input_sg.shape[1]])
            input_sg[:, int(start):int(stop)] = np.min(input_sg)
    return input_sg

# NOISE AND BACKGROUND SIGNAL

# Strenghten/weaken the main signal
def raise_to_power(input_sg, max_power = 2): # max_power= the maximum power to which the pixels of the spektrogram should be raised 
    power = np.random.beta(2,2)*max_power
    output_sg = input_sg**power
    return output_sg

# Add real noise
'''def add_noise(input_sg, noise_sg, noise_to_signal_max_ratio = 10): # noise_to_signal_max_ratio: How much is the noise at most intensified
    # mix with a random noise clip from real data
    ratio = np.random.random()*noise_to_signal_max_ratio
    output_sg = (input_sg + ratio*noise_sg)/(ratio+1)
    return output_sg'''
    
def add_noise(input_sg, noise_level=0.1):
    noise = np.random.normal(loc=0.0, scale=1.0, size=input_sg.shape)  # Gaussian noise
    output_sg = input_sg + noise_level * noise
    output_sg = np.clip(output_sg, a_min=0, a_max=None)  # clip values to original range  
    return output_sg

# Add another vocalization
def add_vocal(input_sg, vocal_sg, beta1=2, beta2=3): # beta1 & beta2: Control the beta distribution to sample the weighting between the spectrograms
    proportion = np.random.beta(beta1,beta2)
    output_sg = (proportion*input_sg + (1-proportion)*vocal_sg)
    return output_sg

# AUGMENTATION FUNCTION
def data_augmentation(input_sg,  
                      p_y_stretch = 0.5, # The probability to apply y_stretch on a single spectrogram
                      y_drop_rate = 0.1, 
                      p_x_stretch = 0.5, # The probability to apply x_stretch on a single spectrogram
                      x_drop_rate = 0.1, 
                      p_y_shift = 0.5, # The probability to apply y_shift on a single spectrogram
                      y_shift_rate = 0.1, 
                      p_x_shift = 0.5, # The probability to apply x_shift on a single spectrogram
                      x_shift_rate = 0.1, 
                      p_power = 0.5, # The probability to apply raise_to_power on a single spectrogram
                      max_power = 2, 
                      p_noise = 0.5, # The probability to apply add_noise on a single spectrogram
                      noise_path = '', # path to noise files
                      noise_to_signal_max_ratio = 10, 
                      noise_level = 0.1, 
                      p_mixup = 0.5, # The probability to apply add_vocal on a single spectrogram                      
                      vocal_path = '', # list of paths to mixup files
                      beta1 = 4, 
                      beta2 = 5, 
                      masking = True, # is masking applied
                      fr_lambda = 0.5, 
                      fr_mask_h = 0.1, 
                      t_lambda = 0.5, 
                      t_mask_l = 0.2):
    mixup_file = "None" # Return infromation of possible mixup with another species
    input_sg = input_sg.copy()
    
    # STRETCH AND SHIFT
    if(np.random.rand() < p_y_stretch):
        input_sg = y_stretch(input_sg, drop_rate = y_drop_rate)
    if(np.random.rand() < p_x_stretch):
        input_sg = x_stretch(input_sg, drop_rate = x_drop_rate)
    if(np.random.rand() < p_y_shift):
        input_sg = y_shift(input_sg, shift_rate = y_shift_rate)
    if(np.random.rand() < p_x_shift):
        input_sg = x_shift(input_sg, shift_rate = x_shift_rate)
    # NOISE AND BACKGROUND SIGNAL
    if(np.random.rand() < p_power):
        input_sg = raise_to_power(input_sg, max_power = max_power) 
    if(np.random.rand() < p_noise):
        #noise = np.load(noise_path + np.random.choice(os.listdir(noise_path)))
        #input_sg = add_noise(input_sg, noise, noise_to_signal_max_ratio = noise_to_signal_max_ratio)
        input_sg = add_noise(input_sg, noise_level = noise_level)
    '''if(np.random.rand() < p_mixup):
        file = random.sample(vocal_path, 1)[0] # randomly choose one path for mixup
        vocal = np.load(file)
        input_sg = add_vocal(input_sg, vocal, beta1, beta2)
        mixup_file = file'''
    # TIME AND FREQUENCY MASKS
    if(masking):    
        mask_direction = np.random.rand()
        if(mask_direction < 0.5):
            input_sg = frequency_mask(input_sg, lamb = fr_lambda, max_height = fr_mask_h)
        else:
            input_sg = time_mask(input_sg, lamb = t_lambda, max_length = t_mask_l)
            
    '''from PIL import Image    
    im = Image.fromarray((np.asarray(input_sg)).astype(np.uint8))
    im.save('aumentada.tiff') 
    '''
    
    return input_sg# , mixup_file
    

