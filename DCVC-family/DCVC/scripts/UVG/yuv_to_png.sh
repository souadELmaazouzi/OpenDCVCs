#!/bin/bash

# Base paths for source and target directories
SRC_DIR="/data/zhan5096/Project/dataset/UVG/cropped_yuv"
OUT_BASE="/data/zhan5096/Project/dataset/UVG/png_sequences"

# Create base output directory if it doesn't exist
mkdir -p "$OUT_BASE"

# Iterate through all YUV files
for yuv_file in "$SRC_DIR"/*.yuv; do
    # Get filename without path and extension
    filename=$(basename "$yuv_file" .yuv)
    
    echo "Processing: $filename"
    
    # Extract resolution information from filename
    # Assuming filename format like Beauty_1920x1024_120fps_420_8bit_YUV.yuv
    resolution=$(echo "$filename" | grep -o "[0-9]\+x[0-9]\+")
    width=$(echo "$resolution" | cut -d'x' -f1)
    height=$(echo "$resolution" | cut -d'x' -f2)
    
    # Create target directory for each sequence
    target_dir="$OUT_BASE/$filename"
    mkdir -p "$target_dir"
    
    # Use ffmpeg to convert YUV to PNG sequence
    ffmpeg -pix_fmt yuv420p -s ${width}x${height} -i "$yuv_file" -f image2 "$target_dir/im%05d.png"
    
    echo "Completed: $filename -> $target_dir"
done

echo "All sequences processed successfully!"