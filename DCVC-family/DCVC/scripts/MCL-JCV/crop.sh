#!/bin/bash

# Define source directory and output directory
SRC_DIR="/home/yichi/Project/dataset/MCL-JVC/sequences"
OUT_DIR="/home/yichi/Project/dataset/MCL-JVC/cropped"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Perform cropping operation on each YUV file
for yuv_file in "$SRC_DIR"/*.yuv; do
    # Extract filename without path
    filename=$(basename "$yuv_file")
    
    # Create new output filename (replace 1080 with 1024)
    output_filename=$(echo "$filename" | sed 's/1080/1024/')
    
    echo "Processing: $filename"
    
    # Use ffmpeg to perform cropping
    ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x1080 -i "$yuv_file" \
           -filter:v "crop=1920:1024:0:0" \
           -f rawvideo -pix_fmt yuv420p "$OUT_DIR/$output_filename"
    
    echo "Completed: $output_filename"
done

echo "All files processed successfully!"