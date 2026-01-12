#!/usr/bin/env python3
"""
Cruise Wing Optimization CLI

순항 익형 최적화 실행 스크립트

Usage:
    # Quick optimization with defaults
    python run_cruise_wing.py
    
    # With custom parameters
    python run_cruise_wing.py --reynolds 5e6 --aoa 4.0 --mach 0.25
    
    # Using scenario file
    python run_cruise_wing.py --scenario scenarios/cruise_wing.yaml
    
    # Direct optimization (no surrogate)
    python run_cruise_wing.py --direct
    
    # Verbose output
    python run_cruise_wing.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cruise_wing.workflow import (
    CruiseWingOptimizer,
    CruiseWingConfig,
    DesignPoint,
    optimize_cruise_wing
)


def main():
    parser = argparse.ArgumentParser(
        description="Cruise Wing Airfoil Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default optimization (Re=3M, α=3°, Ma=0.2)
  python run_cruise_wing.py
  
  # Custom Reynolds number and AoA
  python run_cruise_wing.py --reynolds 5e6 --aoa 4.0
  
  # Use scenario file
  python run_cruise_wing.py --scenario scenarios/cruise_wing.yaml
  
  # Direct optimization without surrogate (faster for few evaluations)
  python run_cruise_wing.py --direct
  
  # Increase training samples for better surrogate
  python run_cruise_wing.py --samples 150
"""
    )
    
    # Design point parameters
    parser.add_argument('--reynolds', '-r', type=float, default=3e6,
                       help='Reynolds number (default: 3e6)')
    parser.add_argument('--aoa', '-a', type=float, default=3.0,
                       help='Angle of attack in degrees (default: 3.0)')
    parser.add_argument('--mach', '-m', type=float, default=0.2,
                       help='Mach number (default: 0.2)')
    
    # Optimization parameters
    parser.add_argument('--direct', action='store_true',
                       help='Use direct XFOIL optimization (no surrogate)')
    parser.add_argument('--samples', '-n', type=int, default=80,
                       help='Number of training samples for surrogate (default: 80)')
    parser.add_argument('--iterations', '-i', type=int, default=50,
                       help='Maximum optimization iterations (default: 50)')
    parser.add_argument('--multistarts', type=int, default=5,
                       help='Number of multi-start optimization runs (default: 5)')
    
    # Constraints
    parser.add_argument('--cl-min', type=float, default=0.4,
                       help='Minimum CL constraint (default: 0.4)')
    parser.add_argument('--ld-min', type=float, default=50.0,
                       help='Minimum L/D constraint (default: 50.0)')
    parser.add_argument('--t-min', type=float, default=0.10,
                       help='Minimum thickness ratio (default: 0.10)')
    
    # Scenario file
    parser.add_argument('--scenario', '-s', type=str,
                       help='Path to YAML scenario file')
    
    # Output
    parser.add_argument('--output', '-o', type=str, 
                       default='output/optimization/cruise_wing',
                       help='Output directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbose = not args.quiet
    if args.verbose:
        verbose = True
    
    try:
        if args.scenario:
            # Use scenario file
            if verbose:
                print(f"Loading scenario: {args.scenario}")
            
            optimizer = CruiseWingOptimizer(scenario_file=args.scenario)
            result = optimizer.run(verbose=verbose)
        else:
            # Use command line parameters
            config = CruiseWingConfig(
                design_points=[DesignPoint(
                    reynolds=args.reynolds,
                    aoa=args.aoa,
                    mach=args.mach,
                    name='cruise'
                )],
                use_surrogate=not args.direct,
                n_training_samples=args.samples,
                max_iterations=args.iterations,
                n_multistart=args.multistarts,
                cl_min=args.cl_min,
                ld_min=args.ld_min,
                t_min=args.t_min,
                output_dir=args.output,
                create_plots=not args.no_plots
            )
            
            optimizer = CruiseWingOptimizer(config=config)
            result = optimizer.run(verbose=verbose)
        
        # Print final results
        if result.success:
            print(f"\n✓ Optimization successful!")
            print(f"  Optimal airfoil: {result.optimal_airfoil['name']}")
            print(f"  L/D: {result.optimal_airfoil.get('L/D', 'N/A'):.2f}")
            return 0
        else:
            print(f"\n✗ Optimization failed")
            return 1
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
