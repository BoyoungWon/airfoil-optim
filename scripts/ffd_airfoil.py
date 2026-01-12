#!/usr/bin/env python3
"""
Free Form Deformation (FFD) Airfoil Generator

This script applies Free Form Deformation to airfoil shapes for parametric
shape optimization and surrogate model generation.

FFD parameterizes the airfoil using a grid of control points, allowing
smooth deformations through Bernstein polynomials.

Usage:
    # Generate FFD-deformed airfoil from NACA baseline
    python ffd_airfoil.py --naca 0012 --control-points 4 3 --deformation 0.0 0.02 0.01

    # Deform existing airfoil
    python ffd_airfoil.py --input naca0012.dat --control-points 5 3 --deformation-file params.txt

    # Generate random samples for surrogate modeling
    python ffd_airfoil.py --naca 0012 --samples 100 --output-dir output/airfoil/ffd_samples
"""

import numpy as np
import subprocess
import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, List, Optional


class FFDAirfoil:
    """
    Free Form Deformation for 2D Airfoil Shapes
    
    Uses Bernstein polynomial-based FFD to deform airfoil coordinates
    by manipulating a lattice of control points.
    """
    
    def __init__(self, n_control_x: int = 5, n_control_y: int = 3):
        """
        Initialize FFD lattice
        
        Parameters:
        -----------
        n_control_x : int
            Number of control points in x-direction (chordwise)
        n_control_y : int
            Number of control points in y-direction (thickness)
        """
        self.n_x = n_control_x
        self.n_y = n_control_y
        
        # Initialize control point lattice
        # Control points span [0, 1] x [y_min, y_max]
        self.control_points = None
        self.bbox = None  # Bounding box: (x_min, x_max, y_min, y_max)
        
    def _bernstein_poly(self, i: int, n: int, u: float) -> float:
        """
        Calculate Bernstein polynomial basis function
        
        B_{i,n}(u) = C(n,i) * u^i * (1-u)^(n-i)
        """
        from math import comb
        return comb(n, i) * (u ** i) * ((1 - u) ** (n - i))
    
    def setup_lattice(self, coords: np.ndarray, padding: float = 0.1):
        """
        Setup FFD control point lattice based on airfoil coordinates
        
        Parameters:
        -----------
        coords : np.ndarray
            Airfoil coordinates (N x 2)
        padding : float
            Padding around airfoil in y-direction (as fraction of thickness)
        """
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Add padding in y-direction
        y_range = y_max - y_min
        y_min -= padding * y_range
        y_max += padding * y_range
        
        self.bbox = (x_min, x_max, y_min, y_max)
        
        # Create uniform control point grid
        x_control = np.linspace(x_min, x_max, self.n_x)
        y_control = np.linspace(y_min, y_max, self.n_y)
        
        # Initialize control points: shape (n_x, n_y, 2)
        self.control_points = np.zeros((self.n_x, self.n_y, 2))
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.control_points[i, j, 0] = x_control[i]
                self.control_points[i, j, 1] = y_control[j]
    
    def _get_local_coords(self, point: np.ndarray) -> Tuple[float, float]:
        """
        Convert physical coordinates to parametric coordinates [0,1] x [0,1]
        """
        x_min, x_max, y_min, y_max = self.bbox
        u = (point[0] - x_min) / (x_max - x_min)
        v = (point[1] - y_min) / (y_max - y_min)
        return u, v
    
    def deform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Deform a single point using FFD
        
        Parameters:
        -----------
        point : np.ndarray
            Original point coordinates [x, y]
        
        Returns:
        --------
        np.ndarray : Deformed point coordinates
        """
        u, v = self._get_local_coords(point)
        
        # Clamp to [0, 1]
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        
        # Calculate deformed position using tensor product of Bernstein polynomials
        new_point = np.zeros(2)
        
        for i in range(self.n_x):
            B_i = self._bernstein_poly(i, self.n_x - 1, u)
            for j in range(self.n_y):
                B_j = self._bernstein_poly(j, self.n_y - 1, v)
                new_point += B_i * B_j * self.control_points[i, j]
        
        return new_point
    
    def deform_airfoil(self, coords: np.ndarray) -> np.ndarray:
        """
        Deform entire airfoil using FFD
        
        Parameters:
        -----------
        coords : np.ndarray
            Original airfoil coordinates (N x 2)
        
        Returns:
        --------
        np.ndarray : Deformed airfoil coordinates (N x 2)
        """
        deformed = np.zeros_like(coords)
        for i, point in enumerate(coords):
            deformed[i] = self.deform_point(point)
        return deformed
    
    def apply_deformation(self, deformation: np.ndarray):
        """
        Apply deformation vector to control points
        
        Parameters:
        -----------
        deformation : np.ndarray
            Deformation vector: can be flat array or (n_x, n_y, 2) shaped
            Typically only y-displacements are used for airfoils
        """
        if deformation.size == self.n_x * self.n_y:
            # Interpret as y-displacements only
            deformation_2d = deformation.reshape(self.n_x, self.n_y)
            for i in range(self.n_x):
                for j in range(self.n_y):
                    self.control_points[i, j, 1] += deformation_2d[i, j]
        elif deformation.shape == (self.n_x, self.n_y, 2):
            self.control_points += deformation
        else:
            raise ValueError(f"Invalid deformation shape: {deformation.shape}")
    
    def get_design_vector(self) -> np.ndarray:
        """
        Get flattened design vector (y-displacements from initial positions)
        
        Returns:
        --------
        np.ndarray : Flattened design vector of y-coordinates
        """
        return self.control_points[:, :, 1].flatten()
    
    def set_design_vector(self, design_vector: np.ndarray, initial_y: np.ndarray):
        """
        Set control points from design vector
        
        Parameters:
        -----------
        design_vector : np.ndarray
            Flattened y-coordinates of control points
        initial_y : np.ndarray
            Initial y-coordinates to compute relative deformation
        """
        y_coords = design_vector.reshape(self.n_x, self.n_y)
        for i in range(self.n_x):
            for j in range(self.n_y):
                self.control_points[i, j, 1] = y_coords[i, j]


def load_airfoil(filepath: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Load airfoil coordinates from file
    
    Returns:
    --------
    tuple : (coordinates, airfoil_name)
    """
    coords = []
    airfoil_name = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    first_line = True
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        # Try to parse as coordinates
        try:
            x, y = float(parts[0]), float(parts[1])
            coords.append([x, y])
        except ValueError:
            # Likely the airfoil name
            if first_line:
                airfoil_name = line
        
        first_line = False
    
    return np.array(coords), airfoil_name


def save_airfoil(filepath: str, coords: np.ndarray, name: str = "FFD Airfoil"):
    """
    Save airfoil coordinates to file in XFOIL format
    """
    with open(filepath, 'w') as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:12.8f}  {y:12.8f}\n")
    print(f"✓ Saved airfoil to: {filepath}")


def generate_naca_baseline(naca_code: str, output_file: str = "temp_baseline.dat", 
                           num_points: int = 160) -> Optional[np.ndarray]:
    """
    Generate NACA airfoil using XFOIL as baseline
    
    Returns:
    --------
    np.ndarray : Airfoil coordinates
    """
    xfoil_commands = f"""PLOP
G

NACA {naca_code}
PPAR
N
{num_points}


SAVE
{output_file}

QUIT
"""
    
    try:
        process = subprocess.run(
            ['xfoil'],
            input=xfoil_commands,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if Path(output_file).exists():
            coords, _ = load_airfoil(output_file)
            return coords
        else:
            print("✗ Failed to generate NACA baseline with XFOIL")
            return None
    except Exception as e:
        print(f"✗ Error generating baseline: {e}")
        return None


def generate_random_deformation(n_control_x: int, n_control_y: int, 
                               amplitude: float = 0.02) -> np.ndarray:
    """
    Generate random deformation vector for sampling
    
    Parameters:
    -----------
    n_control_x, n_control_y : int
        Number of control points
    amplitude : float
        Maximum deformation amplitude (as fraction of chord)
    
    Returns:
    --------
    np.ndarray : Random deformation vector (y-displacements)
    """
    # Use smooth random deformations
    # Don't deform leading/trailing edge control points (i=0, i=n_x-1) as much
    deformation = np.random.uniform(-amplitude, amplitude, (n_control_x, n_control_y))
    
    # Reduce deformation at edges
    deformation[0, :] *= 0.3
    deformation[-1, :] *= 0.3
    
    # Centerline (j = n_y//2) can have more deformation for camber changes
    return deformation


def main():
    parser = argparse.ArgumentParser(
        description="Free Form Deformation Airfoil Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single FFD airfoil from NACA baseline
  python ffd_airfoil.py --naca 0012 --control-points 5 3 --amplitude 0.02 -o output/airfoil/ffd_0012.dat
  
  # Generate multiple samples for surrogate modeling
  python ffd_airfoil.py --naca 2412 --samples 50 --control-points 4 3 --amplitude 0.03 --output-dir output/airfoil/ffd_samples
  
  # Deform existing airfoil
  python ffd_airfoil.py --input custom.dat --control-points 6 3 --amplitude 0.01 -o output/airfoil/custom_ffd.dat
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--naca', type=str,
                           help='NACA 4 or 5-digit code for baseline airfoil')
    input_group.add_argument('--input', type=str,
                           help='Input airfoil coordinate file')
    
    # FFD parameters
    parser.add_argument('--control-points', type=int, nargs=2, default=[5, 3],
                       metavar=('NX', 'NY'),
                       help='Number of control points in x and y directions (default: 5 3)')
    
    # Deformation options
    parser.add_argument('--amplitude', type=float, default=0.02,
                       help='Deformation amplitude as fraction of chord (default: 0.02)')
    parser.add_argument('--deformation', type=float, nargs='+',
                       help='Specific deformation values (flattened array)')
    parser.add_argument('--deformation-file', type=str,
                       help='File containing deformation parameters (one per line)')
    
    # Sampling for surrogate model
    parser.add_argument('--samples', type=int,
                       help='Generate N random samples for surrogate modeling')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str,
                       help='Output airfoil file (required if not using --samples)')
    parser.add_argument('--output-dir', type=str, default='output/airfoil/ffd',
                       help='Output directory for multiple samples (default: output/airfoil/ffd)')
    
    # Visualization
    parser.add_argument('--plot', action='store_true',
                       help='Plot original and deformed airfoils')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.samples and not args.output:
        parser.error("Either --output or --samples must be specified")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load or generate baseline airfoil
    print("=" * 60)
    print("FFD Airfoil Generator")
    print("=" * 60)
    
    if args.naca:
        print(f"\n📐 Generating NACA {args.naca} baseline...")
        baseline_coords = generate_naca_baseline(args.naca)
        baseline_name = f"NACA {args.naca}"
        if baseline_coords is None:
            print("✗ Failed to generate baseline airfoil")
            return 1
    else:
        print(f"\n📁 Loading baseline from: {args.input}")
        if not Path(args.input).exists():
            print(f"✗ Input file not found: {args.input}")
            return 1
        baseline_coords, baseline_name = load_airfoil(args.input)
        baseline_name = baseline_name or Path(args.input).stem
    
    print(f"✓ Loaded baseline: {baseline_name} ({len(baseline_coords)} points)")
    
    # Initialize FFD
    n_x, n_y = args.control_points
    print(f"\n🔧 Initializing FFD with {n_x} x {n_y} control points...")
    ffd = FFDAirfoil(n_control_x=n_x, n_control_y=n_y)
    ffd.setup_lattice(baseline_coords, padding=0.15)
    print(f"✓ FFD lattice setup complete")
    
    # Generate samples or single deformation
    if args.samples:
        print(f"\n🎲 Generating {args.samples} random samples...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also save baseline
        baseline_file = output_dir / f"{baseline_name.replace(' ', '_')}_baseline.dat"
        save_airfoil(str(baseline_file), baseline_coords, baseline_name)
        
        # Save deformation parameters
        param_file = output_dir / "deformation_parameters.txt"
        param_list = []
        
        for i in range(args.samples):
            # Generate random deformation
            deformation = generate_random_deformation(n_x, n_y, args.amplitude)
            
            # Reset FFD and apply deformation
            ffd.setup_lattice(baseline_coords, padding=0.15)
            ffd.apply_deformation(deformation)
            
            # Deform airfoil
            deformed_coords = ffd.deform_airfoil(baseline_coords)
            
            # Save
            output_file = output_dir / f"ffd_sample_{i:04d}.dat"
            name = f"FFD Sample {i:04d}"
            save_airfoil(str(output_file), deformed_coords, name)
            
            # Save parameters
            param_list.append(deformation.flatten())
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{args.samples} samples...")
        
        # Save all parameters
        np.savetxt(param_file, np.array(param_list), 
                  header=f"FFD Deformation Parameters ({n_x}x{n_y} control points, y-displacements)")
        print(f"\n✓ Generated {args.samples} samples in: {output_dir}")
        print(f"✓ Saved parameters to: {param_file}")
        
    else:
        # Single deformation
        if args.deformation:
            print(f"\n🔨 Applying specified deformation...")
            deformation = np.array(args.deformation)
            if deformation.size != n_x * n_y:
                print(f"✗ Deformation size mismatch: expected {n_x * n_y}, got {deformation.size}")
                return 1
            deformation = deformation.reshape(n_x, n_y)
        elif args.deformation_file:
            print(f"\n📄 Loading deformation from: {args.deformation_file}")
            deformation = np.loadtxt(args.deformation_file)
            deformation = deformation.reshape(n_x, n_y)
        else:
            print(f"\n🎲 Generating random deformation (amplitude={args.amplitude})...")
            deformation = generate_random_deformation(n_x, n_y, args.amplitude)
        
        # Apply deformation
        ffd.apply_deformation(deformation)
        
        # Deform airfoil
        print(f"🔄 Deforming airfoil...")
        deformed_coords = ffd.deform_airfoil(baseline_coords)
        
        # Save output
        output_name = f"FFD {baseline_name}"
        save_airfoil(args.output, deformed_coords, output_name)
        print(f"\n✓ Successfully generated FFD airfoil: {args.output}")
    
    # Optional plotting
    if args.plot and not args.samples:
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(baseline_coords[:, 0], baseline_coords[:, 1], 'b-', 
                   linewidth=2, label='Original')
            ax.plot(deformed_coords[:, 0], deformed_coords[:, 1], 'r-', 
                   linewidth=2, label='FFD Deformed')
            
            # Plot control points
            cp_x = ffd.control_points[:, :, 0]
            cp_y = ffd.control_points[:, :, 1]
            ax.plot(cp_x, cp_y, 'go', markersize=8, label='Control Points')
            
            # Draw control lattice
            for i in range(n_x):
                ax.plot(cp_x[i, :], cp_y[i, :], 'g--', alpha=0.3, linewidth=0.5)
            for j in range(n_y):
                ax.plot(cp_x[:, j], cp_y[:, j], 'g--', alpha=0.3, linewidth=0.5)
            
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.set_title(f'FFD Airfoil Deformation ({n_x}x{n_y} control points)')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("\n⚠ matplotlib not available for plotting")
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
