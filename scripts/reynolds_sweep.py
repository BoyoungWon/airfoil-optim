#!/usr/bin/env python3
"""
Reynolds Number Sweep using XFOIL

This script performs a Reynolds number sweep analysis on an airfoil at a fixed AoA.
Runs XFOIL multiple times at different Reynolds numbers.

Usage:
    python reynolds_sweep.py <airfoil_file> <AoA> <Re_min> <Re_max> <dRe>

Example:
    python reynolds_sweep.py naca0012.dat 5.0 100000 5000000 100000
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np


def run_xfoil_single_point(airfoil_path, reynolds, aoa, ncrit=9, iter_limit=100):
    """
    Run XFOIL for a single operating point
    
    Returns:
    --------
    dict : Results dictionary with CL, CD, CDp, CM, Top_Xtr, Bot_Xtr, converged
    """
    
    # Create XFOIL command script
    xfoil_commands = f"""PLOP
G

LOAD
{airfoil_path.absolute()}

OPER
VISC {reynolds}
ITER {iter_limit}
ALFA {aoa}

QUIT
"""
    
    try:
        # Run XFOIL
        process = subprocess.run(
            ['xfoil'],
            input=xfoil_commands,
            text=True,
            capture_output=True,
            timeout=60
        )
        
        output = process.stdout
        
        # Parse output for the final converged results
        # Look for the last occurrence of results in the iteration output
        # Format: "a = 5.000      CL =  0.5560"
        #         "Cm =  0.0022     CD =  0.00848   =>   CDf =  0.00543    CDp =  0.00305"
        
        results = {
            'alpha': aoa,
            'Re': reynolds,
            'CL': None,
            'CD': None,
            'CDp': None,
            'CM': None,
            'Top_Xtr': None,
            'Bot_Xtr': None,
            'converged': False
        }
        
        lines = output.split('\n')
        
        # Find the last converged solution
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            
            # Look for CL in format "a = 5.000      CL =  0.5560"
            if 'a =' in line and 'CL =' in line and results['CL'] is None:
                try:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == 'CL' and j+2 < len(parts):
                            results['CL'] = float(parts[j+2])
                            break
                except (ValueError, IndexError):
                    continue
            
            # Look for CD and CM in format "Cm =  0.0022     CD =  0.00848   =>   CDf =  0.00543    CDp =  0.00305"
            if 'Cm =' in line and 'CD =' in line and results['CD'] is None:
                try:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == 'Cm' and j+2 < len(parts):
                            results['CM'] = float(parts[j+2])
                        elif part == 'CD' and j+2 < len(parts):
                            results['CD'] = float(parts[j+2])
                        elif part == 'CDp' and j+2 < len(parts):
                            results['CDp'] = float(parts[j+2])
                except (ValueError, IndexError):
                    continue
            
            # Look for transition locations
            # Format: "Side 1  free  transition at x/c =  0.1494   44"
            if 'Side 1' in line and 'transition at x/c' in line and results['Top_Xtr'] is None:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        xtr_str = parts[1].split()[0]
                        results['Top_Xtr'] = float(xtr_str)
                except (ValueError, IndexError):
                    pass
            
            if 'Side 2' in line and 'transition at x/c' in line and results['Bot_Xtr'] is None:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        xtr_str = parts[1].split()[0]
                        results['Bot_Xtr'] = float(xtr_str)
                except (ValueError, IndexError):
                    pass
            
            # Check if we have all the data we need
            if (results['CL'] is not None and results['CD'] is not None and 
                results['CM'] is not None and results['CDp'] is not None):
                results['converged'] = True
                break
        
        return results
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"    Error at Re={reynolds:.2e}: {e}")
        return None


def reynolds_sweep(airfoil_file, aoa, re_min, re_max, d_re,
                   output_dir="results/reynolds_sweep", ncrit=9, iter_limit=100):
    """
    Perform Reynolds number sweep analysis
    
    Parameters:
    -----------
    airfoil_file : str
        Path to airfoil coordinate file (.dat)
    aoa : float
        Angle of attack (degrees)
    re_min : float
        Minimum Reynolds number
    re_max : float
        Maximum Reynolds number
    d_re : float
        Reynolds number increment
    output_dir : str
        Directory to save results
    ncrit : float
        Ncrit parameter for transition (default: 9)
    iter_limit : int
        Maximum iterations for viscous solution (default: 100)
    
    Returns:
    --------
    tuple : (results_df, output_files_dict)
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
    
    # Generate Reynolds number array
    # Use logarithmic spacing if the range is large
    re_ratio = re_max / re_min
    if re_ratio > 100:
        # Use log spacing for large ranges
        n_points = int(np.log10(re_max/re_min) / np.log10(1 + d_re/re_min)) + 1
        reynolds_array = np.logspace(np.log10(re_min), np.log10(re_max), n_points)
    else:
        # Use linear spacing
        reynolds_array = np.arange(re_min, re_max + d_re, d_re)
    
    airfoil_name = airfoil_path.stem
    
    print("="*70)
    print("REYNOLDS NUMBER SWEEP ANALYSIS")
    print("="*70)
    print(f"Airfoil:        {airfoil_path.name}")
    print(f"AoA:            {aoa}°")
    print(f"Re range:       {re_min:.2e} to {re_max:.2e}")
    print(f"Number of Re:   {len(reynolds_array)}")
    print(f"Ncrit:          {ncrit}")
    print(f"Iter limit:     {iter_limit}")
    print(f"Output dir:     {output_path}")
    print("="*70)
    
    # Run XFOIL for each Reynolds number
    results = []
    print("\nRunning XFOIL analysis...")
    
    for i, re in enumerate(reynolds_array, 1):
        print(f"  [{i}/{len(reynolds_array)}] Re = {re:.2e}...", end=' ', flush=True)
        
        result = run_xfoil_single_point(airfoil_path, re, aoa, ncrit, iter_limit)
        
        if result and result['converged']:
            results.append(result)
            print(f"✓ CL={result['CL']:.4f}, CD={result['CD']:.6f}")
        else:
            print("✗ Failed to converge")
    
    if not results:
        print("\n✗ No converged results obtained")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_file = output_path / f"{airfoil_name}_aoa{aoa}_Re{re_min:.0e}to{re_max:.0e}.csv"
    txt_file = csv_file.with_suffix('.txt')
    
    df.to_csv(csv_file, index=False)
    
    # Save formatted text file
    with open(txt_file, 'w') as f:
        f.write(f"Reynolds Number Sweep Results\n")
        f.write(f"{'='*70}\n")
        f.write(f"Airfoil: {airfoil_name}\n")
        f.write(f"AoA: {aoa}°\n")
        f.write(f"{'='*70}\n\n")
        f.write(df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total converged points: {len(df)}/{len(reynolds_array)}")
    print(f"\nRe range:   {df['Re'].min():.2e} to {df['Re'].max():.2e}")
    print(f"CL range:   {df['CL'].min():.4f} to {df['CL'].max():.4f}")
    print(f"CD range:   {df['CD'].min():.6f} to {df['CD'].max():.6f}")
    
    if len(df) > 0:
        max_ld_idx = (df['CL']/df['CD']).idxmax()
        print(f"L/D max:    {(df['CL']/df['CD']).max():.2f} at Re={df.loc[max_ld_idx, 'Re']:.2e}")
    
    print(f"\n✓ Results saved:")
    print(f"  CSV: {csv_file}")
    print(f"  TXT: {txt_file}")
    
    output_files = {
        'csv': str(csv_file),
        'txt': str(txt_file)
    }
    
    print(f"{'='*70}")
    
    return df, output_files


def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) < 6:
        print("Usage: python reynolds_sweep.py <AIRFOIL_FILE> <AoA> <Re_min> <Re_max> <dRe> [Ncrit]")
        print()
        print("Parameters:")
        print("  AIRFOIL_FILE  : Path to airfoil coordinate file (.dat)")
        print("  AoA           : Angle of attack (degrees)")
        print("  Re_min        : Minimum Reynolds number")
        print("  Re_max        : Maximum Reynolds number")
        print("  dRe           : Reynolds number increment")
        print("  Ncrit         : Transition criteria (optional, default: 9)")
        print()
        print("Examples:")
        print("  python reynolds_sweep.py naca0012.dat 5.0 100000 5000000 500000")
        print("  python reynolds_sweep.py naca0012.dat 5.0 100000 5000000 500000 9")
        print("  python reynolds_sweep.py custom_airfoil.dat 0.0 50000 1000000 50000 5")
        print("  python reynolds_sweep.py public/airfoil/naca2412.dat 8.0 1e5 1e7 1e5")
        print()
        print("Note: For large Re ranges, logarithmic spacing is automatically used.")
        sys.exit(1)
    
    airfoil_file = sys.argv[1]
    aoa = float(sys.argv[2])
    re_min = float(sys.argv[3])
    re_max = float(sys.argv[4])
    d_re = float(sys.argv[5])
    ncrit = float(sys.argv[6]) if len(sys.argv) > 6 else 9
    
    df, output_files = reynolds_sweep(airfoil_file, aoa, re_min, re_max, d_re, ncrit=ncrit)
    
    if df is not None:
        print(f"\n{'='*70}")
        print("SUCCESS: Reynolds sweep completed")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print("FAILED: Reynolds sweep did not complete successfully")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()
