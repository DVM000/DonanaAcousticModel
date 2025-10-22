# DonanaAcousticModel

Python scripts for bird species classification from audio recordings. This work is focused on species frequently observed in the Doñana Biological Reserve (Spain) - available [here](Data/SpeciesList.txt).

Models were trained with data from [Xeno-Canto](https://xeno-canto.org/) and [iNaturalist Sounds](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ef3713b8d72266e803f9346088fed92d-Abstract-Datasets_and_Benchmarks_Track.html). These recordings were pre-segmented using BirdNET, to collect 3-second segments ready for training:


<img src="Data/pre-processing.png" width="600">


## Code

### Models

Output folder with trained models.  
Model names follow this notation:  
`<CNN>-<resolution>-<n_labels>-<extra>`, where `wi` stands for *weight imprinting*, `ft` for *fine-tuning*, and `q` for *quantized*.  

Reference names:

- `mobilenet-224-337wi-ft.*` → **AVISNet-224**  
- `mobilenet-128-337wi-ft.*` → **AVISNet-128**

Additionally, models trained on a subset of **305 species** are denoted as *AVISNet-224 (305)* and *AVISNet-128 (305)*, corresponding to:

- `mobilenet-224-305.*`  
- `mobilenet-128-305.*`

See more information [here](Models/README-models.md).


### Testing

End-to-end utilities to load a trained model and evaluate it on audio files, producing classification results.

- `analyze-audios.py`: TensorFlow-based script.
- `analyze-audios-tflite.py`: TFLite-based script, for edge inference.

Example execution (see example prediction file `example-predictions.txt`):

```cd Testing```

```python analyze-audios.py --i <audiofolder> --o <predictionsfolder> --min_conf 0.5 --overlap 0```

```python analyze-audios-tflite.py --i <audiofolder> --o <predictionsfolder> --min_conf 0.5 --overlap 0```

```
cat example-predictions.txt 
0.0     3.0     Bubo bubo       1.00    113954.mp3
6.0     9.0     Bubo bubo       1.00    113954.mp3
12.0    15.0    Ardea cinerea   0.97    113954.mp3
15.0    18.0    Bubo bubo       1.00    113954.mp3
```

### Notebooks

- `example-analyze-tflite.ipynb`: example of loading a trained model and evaluate it an audio file, showing classification results.


### Evaluation results

Evaluation was conducted on two different datasets:

**1) Focal Recordings**

Performance metrics on focal recordings from Xeno-Canto and iNaturalist Sounds, using a confidence threshold of 0.5 and considering the top-1 prediction averaged across 3-second windows:

| Model                     | Top-1 Accuracy | F1-score (Macro) |
|--------------------------|----------------|------------------|
| AVISNet-224              | 0.8025         | 0.6958           |
| AVISNet-128              | 0.7680         | 0.6558           |
| BirdNET                  | 0.8623         | 0.7850           |


**2) Soundscapes**

Performance metrics for three different soundscape datasets. Results are reported at both the vocalization-level (or 5-min census-level for Doñana2425) and dataset-level:

| Dataset        | Model       | P (Voc) | R (Voc) | F1 (Voc) | P (Dataset) | R (Dataset) | F1 (Dataset) |
| -------------- | ----------- | ---------------- | ---------------- | ----------------- | ----------- | ----------- | ------------ |
| **Doñana2425** | AVISNet-224 | 0.32             | 0.35             | 0.33              | 0.39        | 0.83        | 0.53         |
|                | AVISNet-128 | 0.29             | 0.29             | 0.29              | 0.42        | 0.84        | 0.56         |
| **WABAD**      | AVISNet-224 | 0.43             | 0.34             | 0.38              | 0.56        | 0.98        | 0.71         |
|                | AVISNet-128 | 0.39             | 0.32             | 0.35              | 0.58        | 0.97        | 0.72         |
| **NIPS4BPlus** | AVISNet-224 | 0.61             | 0.29             | 0.39              | 0.49        | 0.80        | 0.61         |
|                | AVISNet-128 | 0.41             | 0.34             | 0.38              | 0.53        | 0.78        | 0.63         |



