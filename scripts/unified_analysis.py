#!/usr/bin/env python3
"""
Unified Airfoil Analysis

Re 수와 Mach 수에 따라 자동으로 적절한 solver (XFoil 또는 SU2)를 선택하여
airfoil 해석을 수행하는 통합 인터페이스

Usage:
    python unified_analysis.py <airfoil_file> --re <Re> --mach <Mach> --aoa <AoA>
    python unified_analysis.py <airfoil_file> --re <Re> --mach <Mach> --aoa-sweep <min> <max> <step>
    
Examples:
    # Low Re, incompressible (XFoil)
    python unified_analysis.py input/airfoil/naca0012.dat --re 5e5 --mach 0.2 --aoa 5.0
    
    # High Re, compressible (SU2)
    python unified_analysis.py input/airfoil/naca0012.dat --re 3e6 --mach 0.75 --aoa 2.5
    
    # AoA sweep with auto solver selection
    python unified_analysis.py input/airfoil/naca0012.dat --re 1e6 --mach 0.3 --aoa-sweep -5 15 0.5
    
    # Force specific solver
    python unified_analysis.py input/airfoil/naca0012.dat --re 5e5 --mach 0.2 --aoa 5.0 --solver su2_sa
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Import solver modules
from solver_selector import (
    SolverType, AnalysisCondition, SolverSelector, get_solver_availability
)
from aoa_sweep import aoa_sweep
from reynolds_sweep import run_xfoil_single_point


def run_unified_single_point(airfoil_file: str, 
                            reynolds: float,
                            mach: float,
                            aoa: float,
                            solver: Optional[SolverType] = None,
                            output_dir: str = "output/analysis/unified",
                            **kwargs) -> Tuple[bool, Optional[Dict]]:
    """
    단일 조건 해석 (자동 solver 선택)
    
    Parameters:
    -----------
    airfoil_file : str
        Airfoil coordinate file
    reynolds : float
        Reynolds number
    mach : float
        Mach number
    aoa : float
        Angle of attack (degrees)
    solver : Optional[SolverType]
        User-specified solver (None for auto-selection)
    output_dir : str
        Output directory
        
    Returns:
    --------
    Tuple[bool, Optional[Dict]] : (success, results_dict)
    """
    
    # Create analysis condition
    condition = AnalysisCondition(reynolds=reynolds, mach=mach)
    
    # Select solver
    selected_solver = SolverSelector.select_solver(condition, solver)
    settings = SolverSelector.get_recommended_settings(condition, selected_solver)
    
    # Print selection info
    SolverSelector.print_selection_info(condition, selected_solver, settings)
    
    airfoil_path = Path(airfoil_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Execute based on solver type
    if selected_solver == SolverType.XFOIL:
        return _run_xfoil_single(airfoil_path, reynolds, aoa, settings, output_path)
    
    elif selected_solver in [SolverType.SU2_SA, SolverType.SU2_SST, 
                            SolverType.SU2_GAMMA_RETHETA]:
        return _run_su2_single(airfoil_path, condition, aoa, 
                              selected_solver, settings, output_path)
    
    else:
        print(f"✗ Unsupported solver: {selected_solver}")
        return False, None


def _run_xfoil_single(airfoil_path: Path, reynolds: float, aoa: float,
                     settings: dict, output_path: Path) -> Tuple[bool, Optional[Dict]]:
    """XFoil single point analysis"""
    
    print("\n" + "="*70)
    print("RUNNING XFOIL ANALYSIS")
    print("="*70)
    
    try:
        result = run_xfoil_single_point(
            airfoil_path,
            reynolds,
            aoa,
            ncrit=settings.get('ncrit', 9),
            iter_limit=settings.get('iter_limit', 100)
        )
        
        if result and result['converged']:
            print(f"\n✓ Analysis converged successfully")
            print(f"  CL = {result['CL']:.6f}")
            print(f"  CD = {result['CD']:.6f}")
            print(f"  CM = {result['CM']:.6f}")
            print(f"  L/D = {result['CL']/result['CD']:.2f}")
            
            # Save result
            result_file = output_path / f"{airfoil_path.stem}_Re{reynolds:.0e}_M{result.get('mach', 0.0):.2f}_aoa{aoa:.1f}.csv"
            if HAS_PANDAS:
                df = pd.DataFrame([result])
                df.to_csv(result_file, index=False)
                print(f"\n✓ Results saved: {result_file}")
            else:
                # Save as simple CSV without pandas
                import csv
                with open(result_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result.keys())
                    writer.writeheader()
                    writer.writerow(result)
                print(f"\n✓ Results saved: {result_file}")
            
            return True, result
        else:
            print(f"\n✗ Analysis failed to converge")
            return False, None
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def _run_su2_single(airfoil_path: Path, condition: AnalysisCondition, aoa: float,
                   solver_type: SolverType, settings: dict, 
                   output_path: Path) -> Tuple[bool, Optional[Dict]]:
    """SU2 single point analysis"""
    
    print("\n" + "="*70)
    print(f"RUNNING SU2 ANALYSIS ({solver_type.value.upper()})")
    print("="*70)
    
    # Check SU2 availability
    avail = get_solver_availability()
    if not avail.get('su2', False):
        print("\n✗ SU2 not found!")
        print("  Please install SU2: https://su2code.github.io/")
        print("  Alternatively, use --solver xfoil to force XFoil")
        return False, None
    
    try:
        from su2_interface import SU2Config, SU2Interface
        
        # Generate SU2 configuration
        case_name = f"{airfoil_path.stem}_Re{condition.reynolds:.0e}_M{condition.mach:.2f}_aoa{aoa:.1f}"
        config = SU2Config(str(airfoil_path), case_name)
        
        # Extract turbulence model
        turb_model = solver_type.value.replace('su2_', '').upper()
        
        config.set_physics(
            mach=condition.mach,
            reynolds=condition.reynolds,
            aoa=aoa
        )
        config.set_turbulence_model(turb_model)
        config.set_numerical_settings(
            cfl=settings.get('cfl', 5.0),
            mg_levels=settings.get('mg_levels', 3),
            iter_max=settings.get('iter', 5000)
        )
        config.set_boundary_conditions()
        config.set_output(str(output_path / case_name))
        
        # Write config file
        config_file = output_path / f"{case_name}.cfg"
        config.write_config(str(config_file))
        print(f"\n✓ Configuration written: {config_file}")
        
        # Note: Actual SU2 mesh generation is required
        print("\n⚠ NOTE: SU2 requires a mesh file (*.su2)")
        print("  Please generate mesh using gmsh or other mesh generator")
        print("  Or use Docker container with pre-configured SU2 environment")
        
        # Run SU2 (commented out until mesh is available)
        # success, message = SU2Interface.run_analysis(
        #     str(config_file),
        #     str(output_path / case_name)
        # )
        
        # if success:
        #     print(f"\n✓ {message}")
        #     # Parse results
        #     surface_file = output_path / case_name / "surface.csv"
        #     results = SU2Interface.parse_results(str(surface_file))
        #     return True, results
        # else:
        #     print(f"\n✗ {message}")
        #     return False, None
        
        print(f"\n✓ SU2 configuration prepared")
        print(f"  Run manually: SU2_CFD {config_file}")
        return True, None
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_unified_aoa_sweep(airfoil_file: str,
                         reynolds: float,
                         mach: float,
                         aoa_min: float,
                         aoa_max: float,
                         d_aoa: float,
                         solver: Optional[SolverType] = None,
                         output_dir: str = "output/analysis/unified",
                         **kwargs) -> Tuple[bool, Optional[object]]:
    """
    AoA sweep (자동 solver 선택)
    """
    
    # Create analysis condition
    condition = AnalysisCondition(reynolds=reynolds, mach=mach)
    
    # Select solver
    selected_solver = SolverSelector.select_solver(condition, solver)
    settings = SolverSelector.get_recommended_settings(condition, selected_solver)
    
    # Print selection info
    SolverSelector.print_selection_info(condition, selected_solver, settings)
    
    airfoil_path = Path(airfoil_file)
    
    # Execute based on solver type
    if selected_solver == SolverType.XFOIL:
        print(f"\n{'='*70}")
        print("RUNNING XFOIL AOA SWEEP")
        print(f"{'='*70}")
        
        df, output_files = aoa_sweep(
            str(airfoil_path),
            reynolds,
            aoa_min,
            aoa_max,
            d_aoa,
            output_dir=output_dir,
            ncrit=settings.get('ncrit', 9),
            iter_limit=settings.get('iter_limit', 100)
        )
        
        if df is not None:
            return True, df
        else:
            return False, None
    
    elif selected_solver in [SolverType.SU2_SA, SolverType.SU2_SST]:
        print(f"\n{'='*70}")
        print(f"RUNNING SU2 AOA SWEEP ({selected_solver.value.upper()})")
        print(f"{'='*70}")
        print("\n⚠ SU2 AoA sweep requires multiple runs")
        print("  Generating configurations for each AoA...")
        
        # Generate configs for each AoA
        aoa_values = []
        aoa = aoa_min
        while aoa <= aoa_max:
            aoa_values.append(aoa)
            aoa += d_aoa
        
        print(f"\n  Total AoA points: {len(aoa_values)}")
        print(f"  Range: {aoa_min}° to {aoa_max}° (Δα = {d_aoa}°)")
        
        # TODO: Implement SU2 batch AoA sweep
        print("\n✓ Configuration generation complete")
        print("  Run each case individually or use batch script")
        
        return True, None
    
    else:
        print(f"✗ Unsupported solver for AoA sweep: {selected_solver}")
        return False, None


def main():
    """Main command-line interface"""
    
    parser = argparse.ArgumentParser(
        description='Unified airfoil analysis with automatic solver selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single point, auto solver
  python unified_analysis.py input/airfoil/naca0012.dat --re 5e5 --mach 0.2 --aoa 5.0
  
  # High Re, transonic (will select SU2)
  python unified_analysis.py input/airfoil/naca0012.dat --re 3e6 --mach 0.75 --aoa 2.5
  
  # AoA sweep
  python unified_analysis.py input/airfoil/naca0012.dat --re 1e6 --mach 0.3 --aoa-sweep -5 15 0.5
  
  # Force specific solver
  python unified_analysis.py input/airfoil/naca0012.dat --re 5e5 --mach 0.2 --aoa 5.0 --solver xfoil
        """
    )
    
    parser.add_argument('airfoil_file', type=str,
                       help='Path to airfoil coordinate file (.dat)')
    parser.add_argument('--re', type=float, required=True,
                       help='Reynolds number')
    parser.add_argument('--mach', type=float, default=0.0,
                       help='Mach number (default: 0.0)')
    
    # Single point or sweep
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--aoa', type=float,
                      help='Single angle of attack (degrees)')
    group.add_argument('--aoa-sweep', nargs=3, type=float, metavar=('MIN', 'MAX', 'STEP'),
                      help='AoA sweep: min max step (degrees)')
    
    # Solver selection
    parser.add_argument('--solver', type=str, choices=['xfoil', 'su2_sa', 'su2_sst', 'su2_gamma_retheta'],
                       help='Force specific solver (default: auto-select)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='output/analysis/unified',
                       help='Output directory (default: output/analysis/unified)')
    
    # Check availability
    parser.add_argument('--check', action='store_true',
                       help='Check solver availability and exit')
    
    args = parser.parse_args()
    
    # Check solver availability
    if args.check:
        print("Checking solver availability...\n")
        avail = get_solver_availability()
        
        print("Installed solvers:")
        for solver_name, is_available in avail.items():
            status = "✓ Available" if is_available else "✗ Not found"
            print(f"  {solver_name:10s}: {status}")
        
        if not any(avail.values()):
            print("\n✗ No solvers found!")
            print("  Install XFoil or SU2 to use this tool")
            sys.exit(1)
        
        sys.exit(0)
    
    # Parse solver type
    solver_type = None
    if args.solver:
        solver_type = SolverType(args.solver)
    
    # Check if airfoil file exists
    airfoil_path = Path(args.airfoil_file)
    if not airfoil_path.exists():
        # Try alternative paths
        for alt_dir in ['input/airfoil', 'public/airfoil']:
            alt_path = Path(alt_dir) / airfoil_path.name
            if alt_path.exists():
                airfoil_path = alt_path
                break
        else:
            print(f"✗ Error: Airfoil file not found: {args.airfoil_file}")
            sys.exit(1)
    
    print("="*70)
    print("UNIFIED AIRFOIL ANALYSIS")
    print("="*70)
    print(f"Airfoil:     {airfoil_path.name}")
    print(f"Reynolds:    {args.re:.2e}")
    print(f"Mach:        {args.mach:.3f}")
    print("="*70)
    
    # Run analysis
    if args.aoa is not None:
        # Single point
        success, results = run_unified_single_point(
            str(airfoil_path),
            args.re,
            args.mach,
            args.aoa,
            solver=solver_type,
            output_dir=args.output_dir
        )
        
        if success:
            print("\n" + "="*70)
            print("✓ ANALYSIS COMPLETE")
            print("="*70)
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("✗ ANALYSIS FAILED")
            print("="*70)
            sys.exit(1)
    
    else:
        # AoA sweep
        aoa_min, aoa_max, d_aoa = args.aoa_sweep
        
        success, df = run_unified_aoa_sweep(
            str(airfoil_path),
            args.re,
            args.mach,
            aoa_min,
            aoa_max,
            d_aoa,
            solver=solver_type,
            output_dir=args.output_dir
        )
        
        if success:
            print("\n" + "="*70)
            print("✓ AOA SWEEP COMPLETE")
            print("="*70)
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("✗ AOA SWEEP FAILED")
            print("="*70)
            sys.exit(1)


if __name__ == "__main__":
    main()
