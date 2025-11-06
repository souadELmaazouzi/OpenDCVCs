#!/bin/bash

# Define source and output directories
SRC_DIR="/data/zhan5096/Project/dataset/UVG"
OUT_DIR="/data/zhan5096/Project/dataset/UVG/cropped"

# Create the output directory (if it does not exist)
mkdir -p "$OUT_DIR"

# Perform a crop operation on each YUV file
for yuv_file in "$SRC_DIR"/*_1920x1080_*_YUV.yuv; do
    # Extract file name (without path)
    filename=$(basename "$yuv_file")
    
    # Create new output file name (replace 1080 with 1024)
    output_filename=$(echo "$filename" | sed 's/1920x1080/1920x1024/')
    
    echo "Processing: $filename"
    
    # Perform cropping with ffmpeg
    ffmpeg -pix_fmt yuv420p -s 1920x1080 -i "$yuv_file" -vf crop=1920:1024:0:0 "$OUT_DIR/$output_filename"
    
    echo "Done: $output_filename"
done

echo "All sequence processing is complete!"