#!/usr/bin/env python3
"""
Airfoil Import Script using XFOIL

This script imports an airfoil coordinate file, validates it using XFOIL,
and saves it to the public/airfoil directory.

XFOIL supports the following formats:
- Plain coordinate file (just x,y coordinates)
- Labeled coordinate file (airfoil name + coordinates)
- ISES coordinate file
- MSES coordinate file

Usage:
    python import_airfoil.py /path/to/airfoil.dat
    python import_airfoil.py custom_airfoil.dat
"""

import subprocess
import sys
import os
from pathlib import Path
import re


def validate_coordinate_file(file_path):
    """
    Perform basic validation on the coordinate file
    
    Parameters:
    -----------
    file_path : Path
        Path to the airfoil coordinate file
    
    Returns:
    --------
    tuple : (is_valid, message, airfoil_name)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 10:
            return False, "File has too few lines (less than 10)", None
        
        # Remove comment lines (starting with #)
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if len(data_lines) < 10:
            return False, "File has too few data lines after removing comments", None
        
        # Try to detect airfoil name (first line that doesn't start with numbers)
        airfoil_name = None
        coordinate_start_idx = 0
        
        first_line = data_lines[0].strip()
        # Check if first line is a name (not parseable as two numbers)
        try:
            parts = first_line.split()
            if len(parts) >= 2:
                float(parts[0])
                float(parts[1])
                # First line is coordinates
                coordinate_start_idx = 0
                airfoil_name = Path(file_path).stem  # Use filename as name
        except (ValueError, IndexError):
            # First line is likely the airfoil name
            airfoil_name = first_line
            coordinate_start_idx = 1
        
        # Validate that remaining lines are coordinate pairs
        coord_count = 0
        for line in data_lines[coordinate_start_idx:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                # Basic sanity check: airfoil coordinates typically in range
                if abs(x) > 10 or abs(y) > 10:
                    return False, f"Coordinate values seem unreasonable: x={x}, y={y}", None
                coord_count += 1
            except ValueError:
                return False, f"Invalid coordinate line: {line}", None
        
        if coord_count < 10:
            return False, f"Too few valid coordinate pairs ({coord_count})", None
        
        return True, f"Valid coordinate file with {coord_count} points", airfoil_name
    
    except Exception as e:
        return False, f"Error reading file: {e}", None


def import_airfoil(input_file, output_dir="public/airfoil"):
    """
    Import an airfoil coordinate file using XFOIL validation
    
    Parameters:
    -----------
    input_file : str
        Path to the input airfoil coordinate file
    output_dir : str
        Directory to save the validated airfoil file
    
    Returns:
    --------
    str : Path to the saved airfoil file, or None if failed
    """
    
    input_path = Path(input_file)
    
    # Check if file exists
    if not input_path.exists():
        print(f"✗ Error: File not found: {input_file}")
        return None
    
    # Check file extension
    if input_path.suffix.lower() != '.dat':
        print(f"✗ Error: File must have .dat extension (got: {input_path.suffix})")
        print(f"  Hint: Rename your file to {input_path.stem}.dat")
        return None
    
    print(f"Importing airfoil from: {input_path}")
    print(f"File size: {input_path.stat().st_size} bytes")
    
    # Perform basic validation
    print("\nValidating coordinate format...")
    is_valid, message, airfoil_name = validate_coordinate_file(input_path)
    
    if not is_valid:
        print(f"✗ Validation failed: {message}")
        return None
    
    print(f"✓ {message}")
    if airfoil_name:
        print(f"  Airfoil name: {airfoil_name}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    output_file = output_path / input_path.name
    
    # Create XFOIL command script to validate and save
    # XFOIL will reject invalid formats
    xfoil_commands = f"""PLOP
G

LOAD
{input_path.absolute()}

SAVE
{output_file.absolute()}

QUIT
"""
    
    print(f"\nValidating with XFOIL...")
    
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
            print(f"✓ Successfully imported and saved: {output_file}")
            print(f"  Output file size: {output_file.stat().st_size} bytes")
            
            # Display some info from XFOIL output
            if "Buffer airfoil set" in process.stdout:
                # Extract number of points from XFOIL output
                match = re.search(r'using\s+(\d+)\s+points', process.stdout)
                if match:
                    num_points = match.group(1)
                    print(f"  Number of points: {num_points}")
            
            return str(output_file)
        else:
            print(f"✗ Failed to import airfoil")
            print("\nXFOIL output:")
            print(process.stdout)
            if process.stderr:
                print("\nXFOIL errors:")
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


def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) < 2:
        print("Usage: python import_airfoil.py <AIRFOIL_FILE>")
        print()
        print("Examples:")
        print("  python import_airfoil.py /path/to/custom_airfoil.dat")
        print("  python import_airfoil.py my_airfoil.dat")
        print()
        print("Supported formats:")
        print("  - Plain coordinate file (x, y pairs)")
        print("  - Labeled coordinate file (name + x, y pairs)")
        print("  - ISES coordinate file")
        print("  - MSES coordinate file")
        print()
        print("The imported airfoil will be saved to: public/airfoil/")
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = import_airfoil(input_file)
    
    if result:
        print(f"\n{'='*60}")
        print(f"SUCCESS: Airfoil imported to {result}")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"FAILED: Could not import airfoil")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
