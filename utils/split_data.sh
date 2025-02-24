#!/bin/bash

train_ratio=0.8
test_ratio=0.1
val_ratio=0.1

# Ensure sum of ratios is 1
if (( $(echo "$train_ratio + $test_ratio + $val_ratio != 1" | bc -l) )); then
    echo "Error: Ratios do not sum to 1."
    exit 1
fi

# Create necessary directories
mkdir -p ./train/raw ./train/processed ./validation/raw ./validation/processed ./test/raw ./test/processed

# Navigate to data directory
cd stems/ || { echo "Error: Directory 'stems/' not found"; exit 1; }

# Get list of all files
files=(*)

# Shuffle files randomly
shuffled_files=($(printf "%s\n" "${files[@]}" | shuf))

# Calculate split sizes
total_files=${#shuffled_files[@]}
train_size=$(echo "$total_files * $train_ratio" | bc | awk '{print int($1+0.5)}')
test_size=$(echo "$total_files * $test_ratio" | bc | awk '{print int($1+0.5)}')
val_size=$(($total_files - $train_size - $test_size)) # Ensuring all files are assigned

# Move files into respective directories
for i in "${!shuffled_files[@]}"; do
    file="${shuffled_files[$i]}"
    if [ $i -lt $train_size ]; then
        cp "$file" ../train/raw/
    elif [ $i -lt $((train_size + test_size)) ]; then
        cp "$file" ../test/raw/
    else
        cp "$file" ../validation/raw/
    fi
done

echo "Data split completed successfully!"
