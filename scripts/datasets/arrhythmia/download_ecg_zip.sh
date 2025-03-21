#!/bin/bash

# Create directory for the dataset
mkdir -p data/raw/ecg
cd data/raw/ecg

# Download the ZIP file (2.3 GB)
wget -O ecg-arrhythmia-1.0.0.zip https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip

# Unzip the file
unzip ecg-arrhythmia-1.0.0.zip

# Remove the zip file to save space
rm ecg-arrhythmia-1.0.0.zip

echo "Download complete. Files are extracted in data/raw/ecg/" 