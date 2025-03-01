#!/usr/bin/env bash

git clone https://github.com/kahst/BirdNET-Analyzer
pip install -r requirements.txt
cd BirdNET-Analyzer
#git rev-parse  HEAD
git checkout aca89e93c717e34815b5975512fd9cbecfb0e828 

