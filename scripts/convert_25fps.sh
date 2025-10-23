#!/bin/bash

# Convert all .mp4 files in the current directory to 60fps without quality loss

for f in *.mp4; do
    # Skip if no .mp4 files found
    [ -e "$f" ] || { echo "No .mp4 files found."; exit 1; }

   
    mkdir "25fps"
    # Define output filename
    out="./25fps/${f}"

    echo "Processing '$f' → '$out'..."
    ffmpeg -i "$f" -vf "fps=25" -c:v libx264 -crf 0 -preset veryslow -c:a copy "$out"
    echo "Done."
done

echo " All videos processed."

