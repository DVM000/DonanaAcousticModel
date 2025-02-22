1. Segment all training/validation data into 3-second intervals using BirdNET and its corresponding species.
 For species not present in BirdNET, segment all the audio file into 3-second chunks
 execute_birdnet_segment.py -> execute BirdNET analyze + BirdNET segments over all data
 segments_speciesname -> modified script segments.py of BirdNET to save data into species-name folders