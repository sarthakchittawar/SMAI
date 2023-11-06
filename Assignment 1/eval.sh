#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file_path.npy>"
    exit 1
fi

file_path="$1"

if [ ! -f "$file_path" ]; then
    echo "Input file not found."
    exit 1
fi

python eval.py "$file_path"
