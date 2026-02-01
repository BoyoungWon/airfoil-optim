#!/usr/bin/env python3
"""
Angle of Attack (AoA) Sweep using XFOIL

This script performs an AoA sweep analysis on an airfoil at a fixed Reynolds number.
Uses XFOIL's built-in ASEQ (Alpha Sequence) command.

Usage:
    python aoa_sweep.py <airfoil_file> <Re> <AoA_min> <AoA_max> <dAoA>

Example:
    python aoa_sweep.py naca0012.dat 1000000 -5 15 0.5
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np


def parse_polar_file(polar_file):
    """
    Parse XFOIL polar output file
    
    Returns:
    --------
    pandas.DataFrame with columns: alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
    """
    try:
        # Read the file and find the data section
        with open(polar_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (starts with "alpha")
        data_start = None
        for i, line in enumerate(lines):
            if 'alpha' in line.lower() and 'CL' in line:
                data_start = i + 1
                break
        
        if data_start is None:
            return None
        
        # Parse data lines
        data = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    data.append([float(x) for x in parts[:7]])
                except ValueError:
                    continue
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
        return df
    
    except Exception as e:
        print(f"Warning: Could not parse polar file: {e}")
        return None


def aoa_sweep(airfoil_file, reynolds, aoa_min, aoa_max, d_aoa, 
              output_dir="results/aoa_sweep", ncrit=9, iter_limit=100):
    """
    Perform AoA sweep analysis using XFOIL
    
    Parameters:
    -----------
    airfoil_file : str
        Path to airfoil coordinate file (.dat)
    reynolds : float
        Reynolds number
    aoa_min : float
        Minimum angle of attack (degrees)
    aoa_max : float
        Maximum angle of attack (degrees)
    d_aoa : float
        AoA increment (degrees)
    output_dir : str
        Directory to save results
    ncrit : float
        Ncrit parameter for transition (default: 9)
    iter_limit : int
        Maximum iterations for viscous solution (default: 100)
    
    Returns:
    --------
    tuple : (polar_data_df, output_files_dict)
    """
    
    airfoil_path = Path(airfoil_file)
    
    # Check if airfoil file exists
    if not airfoil_path.exists():
        # Try looking in input/airfoil directory first
        alt_path = Path("input/airfoil") / airfoil_path.name
        if alt_path.exists():
            airfoil_path = alt_path
        # Then try public/airfoil directory
        elif (Path("public/airfoil") / airfoil_path.name).exists():
            airfoil_path = Path("public/airfoil") / airfoil_path.name
        else:
            print(f"✗ Error: Airfoil file not found: {airfoil_file}")
            print(f"  Searched in: {airfoil_file}, input/airfoil/, public/airfoil/")
            return None, None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output filenames
    airfoil_name = airfoil_path.stem
    polar_file = output_path / f"{airfoil_name}_Re{reynolds:.0e}_aoa{aoa_min}to{aoa_max}.txt"
    dump_file = output_path / f"{airfoil_name}_Re{reynolds:.0e}_aoa{aoa_min}to{aoa_max}_dump.txt"
    
    print("="*70)
    print("AoA SWEEP ANALYSIS")
    print("="*70)
    print(f"Airfoil:        {airfoil_path.name}")
    print(f"Reynolds:       {reynolds:.2e}")
    print(f"AoA range:      {aoa_min}° to {aoa_max}° (Δα = {d_aoa}°)")
    print(f"Ncrit:          {ncrit}")
    print(f"Iter limit:     {iter_limit}")
    print(f"Output dir:     {output_path}")
    print("="*70)
    
    # Create XFOIL command script
    # Note: NORM command normalizes the airfoil coordinates if needed
    # VPAR N sets ncrit, then blank line returns to main OPER menu
    airfoil_abs = str(airfoil_path.absolute())
    polar_abs = str(polar_file.absolute())
    dump_abs = str(dump_file.absolute())
    
    xfoil_commands = f"""PLOP
G

LOAD {airfoil_abs}
NORM
PANE

OPER
VISC {reynolds}
ITER {iter_limit}
VPAR
N
{ncrit}

PACC
{polar_abs}
{dump_abs}
ASEQ {aoa_min} {aoa_max} {d_aoa}

QUIT
"""
    
    print("\nRunning XFOIL...")
    
    try:
        # Run XFOIL with commands via stdin
        process = subprocess.run(
            ['xfoil'],
            input=xfoil_commands,
            text=True,
            capture_output=True,
            timeout=300  # 5 minutes timeout
        )
        
        # Check if polar file was created
        if polar_file.exists():
            print(f"✓ Polar data saved: {polar_file}")
            
            # Parse polar data
            df = parse_polar_file(polar_file)
            
            if df is not None and len(df) > 0:
                print(f"\n{'='*70}")
                print("RESULTS SUMMARY")
                print(f"{'='*70}")
                print(f"Total points computed: {len(df)}")
                print(f"\nAoA range:  {df['alpha'].min():.2f}° to {df['alpha'].max():.2f}°")
                print(f"CL range:   {df['CL'].min():.4f} to {df['CL'].max():.4f}")
                print(f"CD range:   {df['CD'].min():.6f} to {df['CD'].max():.6f}")
                print(f"L/D max:    {(df['CL']/df['CD']).max():.2f} at α={df.loc[(df['CL']/df['CD']).idxmax(), 'alpha']:.2f}°")
                
                # Save to CSV for easier processing
                csv_file = polar_file.with_suffix('.csv')
                df.to_csv(csv_file, index=False)
                print(f"\n✓ CSV data saved: {csv_file}")
                
                output_files = {
                    'polar': str(polar_file),
                    'csv': str(csv_file),
                    'dump': str(dump_file) if dump_file.exists() else None
                }
                
                print(f"{'='*70}")
                return df, output_files
            else:
                print(f"✗ No data points computed")
                print("\nXFOIL output:")
                print(process.stdout[-2000:])  # Last 2000 chars
                return None, None
        else:
            print(f"✗ Polar file not created")
            print("\nXFOIL output:")
            print(process.stdout)
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"✗ XFOIL execution timed out (>5 minutes)")
        return None, None
    except FileNotFoundError:
        print(f"✗ XFOIL not found. Make sure it's installed and in PATH")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) < 6:
        print("Usage: python aoa_sweep.py <AIRFOIL_FILE> <Re> <AoA_min> <AoA_max> <dAoA> [Ncrit]")
        print()
        print("Parameters:")
        print("  AIRFOIL_FILE  : Path to airfoil coordinate file (.dat)")
        print("  Re            : Reynolds number")
        print("  AoA_min       : Minimum angle of attack (degrees)")
        print("  AoA_max       : Maximum angle of attack (degrees)")
        print("  dAoA          : AoA increment (degrees)")
        print("  Ncrit         : Transition criteria (optional, default: 9)")
        print()
        print("Examples:")
        print("  python aoa_sweep.py naca0012.dat 1000000 -5 15 0.5")
        print("  python aoa_sweep.py naca0012.dat 1000000 -5 15 0.5 9")
        print("  python aoa_sweep.py custom_airfoil.dat 500000 0 20 1.0 5")
        print("  python aoa_sweep.py public/airfoil/naca2412.dat 3e6 -10 25 0.25")
        sys.exit(1)
    
    airfoil_file = sys.argv[1]
    reynolds = float(sys.argv[2])
    aoa_min = float(sys.argv[3])
    aoa_max = float(sys.argv[4])
    d_aoa = float(sys.argv[5])
    ncrit = float(sys.argv[6]) if len(sys.argv) > 6 else 9
    
    df, output_files = aoa_sweep(airfoil_file, reynolds, aoa_min, aoa_max, d_aoa, ncrit=ncrit)
    
    if df is not None:
        print(f"\n{'='*70}")
        print("SUCCESS: AoA sweep completed")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print("FAILED: AoA sweep did not complete successfully")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()
