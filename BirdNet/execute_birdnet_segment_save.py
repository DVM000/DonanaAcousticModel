# -*- coding: utf-8 -*-
"""
Created on March 2025

@author: delia
"""
#encoding='utf-8'

import os
import subprocess

# Definir rutas principales
SUBSET = "train"
BASE_DIR = "/dataset/AUDIOS/TFM/"+SUBSET+"_files/"
OUTPUT_DIR_ANALYZE = "./tfm-external/birdnet-output_"+SUBSET+"/"
OUTPUT_DIR_SEGMENTS = "./tfm-external/birdnet-output_"+SUBSET+"/segments/"

BASE_DIR = "/media/delia/HDD/dataset/AUDIOS/TFM/"+SUBSET+"_files/"
#OUTPUT_DIR_ANALYZE = "birdnet-output_"+SUBSET+"/"

#OUTPUT_DIR_ANALYZE = "./output"
#OUTPUT_DIR_SEGMENTS = "./output/segments"
#OUTPUT_DIR_ANALYZE = "/tfm-external/less_classes/"+SUBSET+"/"
#OUTPUT_DIR_SEGMENTS = "/tfm-external/less_classes/"+SUBSET+"/segments/"


SLIST_DIR = "slist_per_species_v3/"  # Directorio donde se guardan los archivos .txt con los nombres científicos
MIN_CONF = 0.5

# Recorrer todas las subcarpetas en BASE_DIR
for i,scientificname in enumerate(os.listdir(BASE_DIR)):
    subfolder_path = os.path.join(BASE_DIR, scientificname)

    # Verificar si es una carpeta
    if os.path.isdir(subfolder_path):
        print(f"{i}/{len(os.listdir(BASE_DIR))} {scientificname}")
        #Accipiter gentilis Acrocephalus scirpaceus Anas crecca Anser anser Anser fabalis Aquila fasciata Ardea alba Ardea cinerea Ardenna gravis Aythya affinis
        # 'Aegypius monachus'  Burhinus oedicnemus Bubo bubo Bubulcus ibis 
        # Chlidonias leucopterus Circaetus gallicus Circus aeruginosus Columba palumbus Curruca communis Cygnus columbianus Cercotrichas galactotes Crex crex
        #if not 'Falco' in scientificname and not 'Larus' in scientificname and not 'Emberiza' in scientificname: continue
        #if not 'Larus' in scientificname and not 'Emberiza' in scientificname: continue
        #if not 'Tachymarptis' in scientificname: continue
        #if 'A' in scientificname or 'B' in scientificname or 'C' in scientificname or 'Larus' in scientificname or 'Emberiza' in scientificname: continue 
        if os.path.isdir(OUTPUT_DIR_SEGMENTS + scientificname):
            print(' --> skipping ', scientificname)
            continue
            
        print(f"Procesando: {scientificname}")
        
        # Ejecutar birdnet_analyzer.analyze
        analyze_command = [
            "python3",
            "-m",
            "birdnet_analyzer.analyze_savepredictions",
            "--i",
            subfolder_path,
            "--o",
            OUTPUT_DIR_ANALYZE,
            "--slist",
            f"{SLIST_DIR}{scientificname}.txt",
            "--min_conf", str(MIN_CONF), 
            "--locale","en",
            # "--max_segments", "1000" #NO allowed
        ]  
        print(analyze_command)    
        try:
            subprocess.run(analyze_command, check=True)
            print(f"Procesado (analyze) {scientificname} con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"Error al procesar (analyze) {scientificname}: {e}")
        
        # Ejecutar birdnet_analyzer.segments_speciesname
        segments_command = [
            "python3",
            "-m",
            "birdnet_analyzer.segments_speciesname",
            "--audio",
            subfolder_path, #.replace(' ', '\ '),
            "--results",
            OUTPUT_DIR_ANALYZE+'txt/',
            "--o",
            OUTPUT_DIR_SEGMENTS,
            "--max_segments", "1000"
        ]
        try:
            subprocess.run(segments_command, check=True)
            print(f"Procesado (segments) {scientificname} con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"Error al procesar (segments) {scientificname}: {e}")
        print(segments_command)
           
     

