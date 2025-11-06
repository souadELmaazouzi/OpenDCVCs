#!/bin/bash

# Base paths to source and target directories
SRC_DIR="/dataset/HEVC-B/cropped"
OUT_BASE="/dataset/HEVC-B/png_sequences"

# Create the base directory for the output (if it does not exist)
mkdir -p "$OUT_BASE"

# Iterate over all YUV files
for yuv_file in "$SRC_DIR"/*.yuv; do
 # Get the filename (without path and extension)
    filename=$(basename "$yuv_file" .yuv)
    
    echo "Processing: $filename"
    
    # Extract resolution information from filename
    # HEVC-B filename format similar to BasketballDrive_1920x1024_50.yuv
    resolution=$(echo "$filename" | grep -o "[0-9]\+x[0-9]\+")
    width=$(echo "$resolution" | cut -d'x' -f1)
    height=$(echo "$resolution" | cut -d'x' -f2)
    
    # Extract frame rate information    
    fps=$(echo "$filename" | grep -o "[0-9]\+$")
    
    # Create a destination directory for each sequence
    target_dir="$OUT_BASE/$filename"
    mkdir -p "$target_dir"
    
    # Convert YUV to PNG sequences using ffmpeg
    # For HEVC-B sequences, explicitly specify yuv420p format and frame rate
    ffmpeg -f rawvideo -pix_fmt yuv420p -s ${width}x${height} -r $fps -i "$yuv_file" \
           -f image2 "$target_dir/im%05d.png"
    
    echo "Done: $filename -> $target_dir"
done

echo "All sequence processing is complete!"