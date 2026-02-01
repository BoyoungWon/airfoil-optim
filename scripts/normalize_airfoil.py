#!/usr/bin/env python3
"""
Normalize airfoil coordinates

Converts airfoil coordinates to normalized form (chord = 1.0)
"""

import sys
import numpy as np
from pathlib import Path

def normalize_airfoil(input_file, output_file=None):
    """
    Normalize airfoil coordinates to chord length = 1.0
    
    Parameters:
    -----------
    input_file : str
        Input airfoil file
    output_file : str, optional
        Output file (default: input_file with _normalized suffix)
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
    else:
        output_path = Path(output_file)
    
    # Read file
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # First line is name
    name = lines[0].strip()
    
    # Check if second line contains point counts
    try:
        parts = lines[1].split()
        if len(parts) == 2:
            # Lednicer format with point counts
            n_upper = int(float(parts[0]))
            n_lower = int(float(parts[1]))
            data_start = 2
            print(f"Lednicer format detected: {n_upper} upper, {n_lower} lower points")
        else:
            # Standard format
            data_start = 1
            print("Standard format detected")
    except:
        data_start = 1
    
    # Read coordinates
    coords = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append([x, y])
            except ValueError:
                continue
    
    coords = np.array(coords)
    
    if len(coords) == 0:
        print("✗ No valid coordinates found!")
        return False
    
    print(f"\nOriginal coordinates:")
    print(f"  Points: {len(coords)}")
    print(f"  X range: {coords[:, 0].min():.4f} to {coords[:, 0].max():.4f}")
    print(f"  Y range: {coords[:, 1].min():.4f} to {coords[:, 1].max():.4f}")
    
    # Calculate chord length
    chord = coords[:, 0].max() - coords[:, 0].min()
    x_min = coords[:, 0].min()
    
    print(f"  Chord length: {chord:.4f}")
    print(f"  X offset: {x_min:.4f}")
    
    # Normalize
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - x_min) / chord
    coords_norm[:, 1] = coords[:, 1] / chord
    
    print(f"\nNormalized coordinates:")
    print(f"  X range: {coords_norm[:, 0].min():.6f} to {coords_norm[:, 0].max():.6f}")
    print(f"  Y range: {coords_norm[:, 1].min():.6f} to {coords_norm[:, 1].max():.6f}")
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(name + '\n')
        for x, y in coords_norm:
            f.write(f"  {x:.6f}  {y:.6f}\n")
    
    print(f"\n✓ Normalized airfoil saved: {output_path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python normalize_airfoil.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    normalize_airfoil(input_file, output_file)
