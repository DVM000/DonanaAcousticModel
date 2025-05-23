Segment all training/validation data into 3-second intervals using BirdNET and its corresponding species.
Note: some species are not present in BirdNET
See ../BirdNET forlder for BirdNET-based segmentation:
- execute_birdnet_segment_save.py -> execute BirdNET analyze + BirdNET segments over all data
- analyze_savepredictions.py -> modified script analyze.py of BirdNET to save logits along with .txt table
- segments_speciesname.py -> modified script segments.py of BirdNET to save data into species-name folders
