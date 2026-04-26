import os
import numpy as np
from sklearn.model_selection import train_test_split

# Output directory where the final split datasets will be saved
OUTPUT_DIR = './dataset_multizone_final'

# List of source zones: each entry is (zone name, path to X array, path to Y array)
SOURCES = [
    ('Golfe_Mexique', './X_Golfe_Mexique.npy', './Y_Golfe_Mexique.npy'),
    ('Cote_Est_USA',  './X_Cote_Est_USA.npy',   './Y_Cote_Est_USA.npy'),
    ('Caraibes',      './X_Caraibes.npy',         './Y_Caraibes.npy'),
]

print('Loading 3 datasets ...')

# Load and collect arrays from all zones
all_X, all_Y = [], []
for zone, x_path, y_path in SOURCES:
    X = np.load(x_path)
    Y = np.load(y_path)
    print(f'   {zone:15s}  X={X.shape}  Y={Y.shape}')
    all_X.append(X)
    all_Y.append(Y)

# Concatenate all zones into a single dataset along the sample axis
X = np.concatenate(all_X, axis=0)
Y = np.concatenate(all_Y, axis=0)
print(f'\nTotal: {X.shape[0]:,} patches -- shape {X.shape}')

# First split: reserve 15% for the test set, keep 85% for train+val
X_tv, X_test, y_tv, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

# Second split: from the remaining 85%, reserve ~15% for validation
# 0.176 * 0.85 = ~0.15 of the full dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.176, random_state=42
)

# Create the output directory if it does not already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save each split to disk and report its size as a percentage of the full dataset
for split, Xs, Ys in [
    ('train', X_train, y_train),
    ('val',   X_val,   y_val),
    ('test',  X_test,  y_test),
]:
    np.save(f'{OUTPUT_DIR}/X_{split}.npy', Xs)
    np.save(f'{OUTPUT_DIR}/y_{split}.npy', Ys)
    print(f'   {split:5s} -> {Xs.shape[0]:>7,} patches  ({100 * Xs.shape[0] / X.shape[0]:.1f}%)')

print(f'\nFinal dataset saved to {OUTPUT_DIR}/')
print('   X_train.npy  y_train.npy  |  X_val.npy  y_val.npy  |  X_test.npy  y_test.npy')
print('   X format: (N, 128, 128, 12) float32  --  Y format: (N, 128, 128) uint8')