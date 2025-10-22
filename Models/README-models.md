
## Models


Model names follow this notation:  
`<CNN>-<resolution>-<n_labels>-<extra>`,  where `wi` stands for *weight imprinting*, `ft` for *fine-tuning*, and `q` for *quantized*.  

Reference names:

- `mobilenet-224-337wi-ft.*` → **AVISNet-224**  
- `mobilenet-128-337wi-ft.*` → **AVISNet-128**


When runnning inference, please select the corresponding species-list file, according to number of model classes:

| filename                 | Model              | Features     |
|--------------------------|--------------------|------------------|
|  mobilenet-128-305.*     | AVISNet-128 (305)  | 128x128. 305 classes  |
|  mobilenet-128-305-q.*   | AVISNet-128-q (305)  | 128x128. 305 classes. Quantization  |
|  **mobilenet-128-337-wi-ft.*** | AVISNet-128      | 128x128. 337 classes  |
|  mobilenet-128-337-wi-ft-q.* | AVISNet-128-q    | 128x128. 337 classes. Quantization  |
|  mobilenet-224-305.*     | AVISNet-224 (305)  | 224x224. 305 classes  |
|  mobilenet-224-305-q.*   | AVISNet-224-q (305)  | 224x224. 305 classes. Quantization  |
|  **mobilenet-224-337-wi-ft.*** | AVISNet-224      | 224x224. 337 classes  |
|  mobilenet-224-337-wi-ft-q.* | AVISNet-224-q    | 224x224. 337 classes. Quantization  |
|  **species-list-305.txt** | Especies list         | 305 classes  |
|  **species-list-337.txt** | Especies list         | 337 classes  |





