#!/bin/bash

# Create directory for the dataset
mkdir -p data/raw/ecg
cd data/raw/ecg

# Download the dataset using wget
# https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
wget -N -c -np https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip

# Unzip and remove the zip file
unzip a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
rm a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip

echo "Download complete. Files are in data/raw/ecg/physionet.org/files/ecg-arrhythmia/1.0.0/"
