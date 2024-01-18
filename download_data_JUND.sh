#!/bin/bash

# URL of the zip file
url="https://www.cs.rpi.edu/~zaki/MLIB/data/TF_data.zip"

# Directory to unpack into
target_directory="./data/JUND"

# Create the target directory if it doesn't exist
mkdir -p "$target_directory"

# Download the zip file
wget "$url" -O "$target_directory/TF_data.zip"

# Unpack the zip file
unzip "$target_directory/TF_data.zip" -d "$target_directory"

# Remove the downloaded zip file (optional)
rm "$target_directory/TF_data.zip"

echo "Download and unpack completed successfully!"
