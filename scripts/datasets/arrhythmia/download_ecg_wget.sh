#!/bin/bash

# Create directory for the dataset
mkdir -p data/raw/ecg
cd data/raw/ecg

# Download the dataset using wget recursively
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/

echo "Download complete. Files are in data/raw/ecg/physionet.org/files/ecg-arrhythmia/1.0.0/" 