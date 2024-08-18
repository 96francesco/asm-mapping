#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/image/directory"
    exit 1
fi

image_dir="$1"

if [ ! -d "$image_dir" ]; then
    echo "Error: $image_dir is not a valid directory"
    exit 1
fi

for file in "$image_dir"/*.tif; do
    width=$(/usr/bin/gdalinfo "$file" | grep "Size is" | awk '{print $3}' | sed 's/,//')
    height=$(/usr/bin/gdalinfo "$file" | grep "Size is" | awk '{print $4}')
    
    x_offset=$(( ($width - 352) / 2 ))
    y_offset=$(( ($height - 352) / 2 ))
    
    /usr/bin/gdal_translate -srcwin $x_offset $y_offset 352 352 "$file" "${file%.tif}_temp.tif"
    mv "${file%.tif}_temp.tif" "$file"
done