#!/bin/bash

# Define source and output directories
SRC_DIR="/dataset/HEVC-B/sequences"
OUT_DIR="/dataset/HEVC-B/cropped"

# Create the output directory (if it does not exist)
mkdir -p "$OUT_DIR"

# Perform a crop operation on each YUV file
for yuv_file in "$SRC_DIR"/*.yuv; do
# Extract file name (without path)
    filename=$(basename "$yuv_file")
    
# Create new output file name (replace 1080 with 1024)
    output_filename=$(echo "$filename" | sed 's/1080/1024/')
    
    echo "Processing: $filename"
    
# Perform cropping with ffmpeg
    ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x1080 -i "$yuv_file" \
           -filter:v "crop=1920:1024:0:0" \
           -f rawvideo -pix_fmt yuv420p "$OUT_DIR/$output_filename"
    
    echo "Done: $output_filename"
done

echo "All sequence processing is complete!"