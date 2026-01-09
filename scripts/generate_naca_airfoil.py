#!/usr/bin/env python3
"""
NACA Airfoil Generator using XFOIL

This script generates NACA 4-digit and 5-digit series airfoil coordinate files
using XFOIL and saves them to the public/airfoil directory.

Usage:
    python generate_naca_airfoil.py 0012
    python generate_naca_airfoil.py 2412
    python generate_naca_airfoil.py 23012
"""

import subprocess
import sys
import os
from pathlib import Path


def generate_naca_airfoil(naca_code, output_dir="public/airfoil", num_points=160):
    """
    Generate NACA airfoil coordinates using XFOIL
    
    Parameters:
    -----------
    naca_code : str or int
        NACA 4-digit or 5-digit series code (e.g., "0012", "2412", "23012")
    output_dir : str
        Directory to save the airfoil coordinate file
    num_points : int
        Number of panel points for airfoil discretization (default: 160)
    
    Returns:
    --------
    str : Path to the generated airfoil file
    """
    
    # Convert to string and validate
    naca_code = str(naca_code).strip()
    
    if not naca_code.isdigit():
        raise ValueError(f"NACA code must be numeric: {naca_code}")
    
    if len(naca_code) not in [4, 5]:
        raise ValueError(f"NACA code must be 4 or 5 digits: {naca_code}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    output_file = output_path / f"naca{naca_code}.dat"
    
    # Create XFOIL command script
    # PLOP G command disables graphics to avoid X11 display requirement
    xfoil_commands = f"""PLOP
G

NACA {naca_code}
PPAR
N
{num_points}


SAVE
{output_file.absolute()}

QUIT
"""
    
    print(f"Generating NACA {naca_code} airfoil...")
    print(f"Output: {output_file.absolute()}")
    
    try:
        # Run XFOIL with commands via stdin
        process = subprocess.run(
            ['xfoil'],
            input=xfoil_commands,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        # Check if file was created
        if output_file.exists():
            print(f"✓ Successfully generated: {output_file}")
            print(f"  File size: {output_file.stat().st_size} bytes")
            return str(output_file)
        else:
            print(f"✗ Failed to generate airfoil file")
            print("XFOIL stdout:")
            print(process.stdout)
            print("XFOIL stderr:")
            print(process.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"✗ XFOIL execution timed out")
        return None
    except FileNotFoundError:
        print(f"✗ XFOIL not found. Make sure it's installed and in PATH")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def batch_generate_common_naca_airfoils(output_dir="public/airfoil"):
    """
    Generate a collection of commonly used NACA airfoils
    """
    common_airfoils = [
        "0006", "0009", "0012", "0015", "0018", "0021",  # Symmetric
        "2412", "2415", "4412", "4415",                  # 4-digit cambered
        "23012", "23015",                                 # 5-digit
    ]
    
    print(f"Generating {len(common_airfoils)} common NACA airfoils...")
    print(f"Output directory: {Path(output_dir).absolute()}")
    print("-" * 60)
    
    results = []
    for naca in common_airfoils:
        result = generate_naca_airfoil(naca, output_dir)
        results.append((naca, result is not None))
        print()
    
    print("-" * 60)
    print(f"Summary: {sum(r[1] for r in results)}/{len(results)} airfoils generated successfully")
    
    return results


def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) < 2:
        print("Usage: python generate_naca_airfoil.py <NACA_CODE> [NUM_POINTS] [OUTPUT_DIR]")
        print("   or: python generate_naca_airfoil.py --batch")
        print()
        print("Examples:")
        print("  python generate_naca_airfoil.py 0012")
        print("  python generate_naca_airfoil.py 0012 200")
        print("  python generate_naca_airfoil.py 2412 160 custom/output")
        print("  python generate_naca_airfoil.py --batch")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        batch_generate_common_naca_airfoils()
    else:
        naca_code = sys.argv[1]
        num_points = int(sys.argv[2]) if len(sys.argv) > 2 else 160
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "public/airfoil"
        generate_naca_airfoil(naca_code, output_dir, num_points)


if __name__ == "__main__":
    main()
