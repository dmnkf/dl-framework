#!/bin/bash

# Check if the directory exists
if [ ! -d "data/raw/ecg/arrhythmia/" ]; then
    echo "Error: Dataset directory not found. Please download the complete dataset and place it in data/raw/arrhythmia/files/ecg-arrhythmia/1.0.0/WFDBRecords"
    exit 1
fi

# Download the CSV file
wget -O data/raw/ecg/arrhythmia/Chapman_Ningbo_ECG_DB_Labeling_Info.csv https://raw.githubusercontent.com/JaeBinCHA7/ECG-Multi-Label-Classification-Using-Multi-Model/main/data_info/Chapman_Ningbo_ECG_DB_Labeling_Info.csv