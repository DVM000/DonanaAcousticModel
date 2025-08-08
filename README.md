# DonanaAcousticModel

Python scripts for bird species classification from audio recordings. This work is focused on species frequently observed in the Doñana Biological Reserve (Spain) - available [here](Data/SpeciesList.txt).

Models were trained with data from [Xeno-Canto](https://xeno-canto.org/) and [iNaturalist Sounds](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ef3713b8d72266e803f9346088fed92d-Abstract-Datasets_and_Benchmarks_Track.html). These recordings were pre-segmented using BirdNET, to collect 3-second segments ready for training:


<img src="Data/pre-processing.png" width="600">


## Code

### Models

Output folder with trained models. Model names follow this notation: `<CNN>-<resolution>-<n_labels>-<extra>`, where `wi` stands for *weight imprinting*, `ft` for *fine-tuning* and `q` for *quantized*. 

Evaluation was conducted on a test set of *focal recordings* from Xeno-Canto and iNaturalist Sounds using a confidence threshold of 0.5 and considering the top-1 prediction averaged across 3-second windows. Performance metrics for the 337-species models are reported below:

| Model                     | Top-1 Accuracy | Balanced Accuracy | F1-score (Macro) | # Files After Thresholding |
|--------------------------|----------------|-------------------|------------------|-----------------------------|
| mobilenet-224-337wi-ft.h5| 0.8025         | 0.7080            | 0.6958           | 45,339                      |
| mobilenet-128-337wi-ft.h5| 0.7680         | 0.6603            | 0.6558           | 45,023                      |
| BirdNET                  | 0.8623         | 0.7718            | 0.7850           | 45,970                      |


### Data

Species list employed in this work.

### Testing

- `analyze-audios.py`: end-to-end utility to load a trained model and evaluate it on audio files, producing classification results.

Example execution (see example prediction file `example-predictions.txt`):

```python analyze-audios.py --i <audiofolder> --o <predictionsfolder> --min_conf 0.5 --overlap 0```


```
cat example-predictions.txt 
0.0     3.0     Bubo bubo       1.00    113954.mp3
6.0     9.0     Bubo bubo       1.00    113954.mp3
12.0    15.0    Ardea cinerea   0.97    113954.mp3
15.0    18.0    Bubo bubo       1.00    113954.mp3
```

### Embedded inference

- `analyze-audios-tflite.py`: end-to-end utility to load a trained model in TensorFLow Lite format, producing classification results.

Same prediction file is produced (`example-predictions.txt`)

### Notebooks

- `example-analyze-tflite.ipynb`: example of loading a trained model and evaluate it an audio file, showing classification results.

