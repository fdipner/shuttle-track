#!/bin/zsh

# Directory containing your input videos
VIDEO_DIR=~/github/badminton-perception/data/phone_20251015

# Output directory
SAVE_DIR=~/github/badminton-perception/results/tracknet

# Model checkpoints
TRACKNET=ckpts/TrackNet_best.pt
INPAINTNET=ckpts/InpaintNet_best.pt

# Loop through all .mp4 files in VIDEO_DIR
for file in "$VIDEO_DIR"/*.mp4; do
  # Skip if no mp4 files found
  [[ -e "$file" ]] || continue

  echo "Processing: $file"
  
  python predict.py \
    --video_file "$file" \
    --tracknet_file "$TRACKNET" \
    --inpaintnet_file "$INPAINTNET" \
    --save_dir "$SAVE_DIR" \
    --output_video \
    --batch_size 1
done

