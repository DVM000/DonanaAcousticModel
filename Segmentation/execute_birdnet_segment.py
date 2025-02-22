# -*- coding: utf-8 -*-
"""
Created on Dec 2024

@author: delia
"""
#encoding='utf-8'

import os
import subprocess

# Definir rutas principales
SUBSET = "test"
BASE_DIR = "/dataset/AUDIOS/TFM/"+SUBSET+"_files/"
#OUTPUT_DIR_ANALYZE = "./val-txt"
#OUTPUT_DIR_SEGMENTS = "./val-segments"
OUTPUT_DIR_ANALYZE = "/tfm-external/birdnet-txt_"+SUBSET+"/"
OUTPUT_DIR_SEGMENTS = "/tfm-external/segments_"+SUBSET+"/"
SLIST_DIR = "slist_per_species/"  # Directorio donde se guardan los archivos .txt con los nombres científicos
MIN_CONF = 0.5

# Recorrer todas las subcarpetas en BASE_DIR
for scientificname in os.listdir(BASE_DIR):
    subfolder_path = os.path.join(BASE_DIR, scientificname)

    # Verificar si es una carpeta
    if os.path.isdir(subfolder_path):
        if not 'Falco' in scientificname and not 'Larus' in scientificname: continue
        print(f"Procesando: {scientificname}")

        # Ejecutar birdnet_analyzer.analyze
        analyze_command = [
            "python3",
            "-m",
            "birdnet_analyzer.analyze",
            "--i",
            subfolder_path,
            "--o",
            OUTPUT_DIR_ANALYZE,
            "--slist",
            f"../{SLIST_DIR}{scientificname}.txt",
            "--min_conf", str(MIN_CONF), 
            "--locale","en"
        ]      
        try:
            subprocess.run(analyze_command, check=True)
            print(f"Procesado (analyze) {scientificname} con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"Error al procesar (analyze) {scientificname}: {e}")


        # Ejecutar birdnet_analyzer.segments_
        segments_command = [
            "python3",
            "-m",
            "birdnet_analyzer.segments_speciesname",
            "--audio",
            subfolder_path, #.replace(' ', '\ '),
            "--results",
            OUTPUT_DIR_ANALYZE,
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

