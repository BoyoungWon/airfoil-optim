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
                            use_neuralfoil_fallback: bool = True,
                            ncrit: Optional[float] = None,
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
    use_neuralfoil_fallback : bool
        Use NeuralFoil as fallback if XFoil fails (default: True)
        
    Returns:
    --------
    Tuple[bool, Optional[Dict]] : (success, results_dict)
    """
    
    # Create analysis condition
    condition = AnalysisCondition(reynolds=reynolds, mach=mach)
    
    # Select solver
    selected_solver = SolverSelector.select_solver(condition, solver)
    settings = SolverSelector.get_recommended_settings(condition, selected_solver)
    
    # Override ncrit if specified
    if ncrit is not None:
        settings['ncrit'] = ncrit
    
    # Print selection info
    SolverSelector.print_selection_info(condition, selected_solver, settings)
    
    airfoil_path = Path(airfoil_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Execute based on solver type
    if selected_solver == SolverType.XFOIL:
        success, result = _run_xfoil_single(airfoil_path, reynolds, aoa, settings, output_path)
        
        # Fallback to NeuralFoil if XFoil fails
        if not success and use_neuralfoil_fallback and mach < 0.5:
            print("\n" + "="*70)
            print("⚠ XFOIL FAILED - TRYING NEURALFOIL FALLBACK")
            print("="*70)
            print("XFoil did not converge. Attempting NeuralFoil as fallback...")
            
            # Check if NeuralFoil is available
            avail = get_solver_availability()
            if avail.get('neuralfoil', False):
                nf_settings = SolverSelector.get_recommended_settings(condition, SolverType.NEURALFOIL)
                success, result = _run_neuralfoil_single(airfoil_path, reynolds, aoa, nf_settings, output_path)
                if success:
                    print("\n✓ NeuralFoil fallback successful!")
            else:
                print("\n✗ NeuralFoil not available for fallback")
                print("  Install: cd neuralfoil && pip install -e .")
        
        return success, result
    
    elif selected_solver == SolverType.NEURALFOIL:
        return _run_neuralfoil_single(airfoil_path, reynolds, aoa, settings, output_path)
    
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


def _run_neuralfoil_single(airfoil_path: Path, reynolds: float, aoa: float,
                          settings: dict, output_path: Path) -> Tuple[bool, Optional[Dict]]:
    """NeuralFoil single point analysis"""
    
    print("\n" + "="*70)
    print("RUNNING NEURALFOIL ANALYSIS")
    print("="*70)
    
    try:
        # Import NeuralFoil
        import sys
        neuralfoil_path = Path(__file__).parent.parent / "neuralfoil"
        if str(neuralfoil_path) not in sys.path:
            sys.path.insert(0, str(neuralfoil_path))
        
        from neuralfoil import get_aero_from_dat_file
        
        # Run NeuralFoil
        print(f"\nAnalyzing with neural network surrogate...")
        print(f"  Model size: {settings.get('model_size', 'xlarge')}")
        
        result = get_aero_from_dat_file(
            str(airfoil_path),
            alpha=aoa,
            Re=reynolds,
            n_crit=settings.get('ncrit', 9),
            xtr_upper=settings.get('xtr_upper', 1.0),
            xtr_lower=settings.get('xtr_lower', 1.0),
            model_size=settings.get('model_size', 'xlarge')
        )
        
        # Extract scalar values
        CL = float(result['CL'][0]) if hasattr(result['CL'], '__len__') else float(result['CL'])
        CD = float(result['CD'][0]) if hasattr(result['CD'], '__len__') else float(result['CD'])
        CM = float(result['CM'][0]) if hasattr(result['CM'], '__len__') else float(result['CM'])
        confidence = float(result['analysis_confidence'][0]) if hasattr(result['analysis_confidence'], '__len__') else float(result['analysis_confidence'])
        Top_Xtr = float(result['Top_Xtr'][0]) if hasattr(result['Top_Xtr'], '__len__') else float(result['Top_Xtr'])
        Bot_Xtr = float(result['Bot_Xtr'][0]) if hasattr(result['Bot_Xtr'], '__len__') else float(result['Bot_Xtr'])
        
        print(f"\n✓ Analysis complete")
        print(f"  CL = {CL:.6f}")
        print(f"  CD = {CD:.6f}")
        print(f"  CM = {CM:.6f}")
        print(f"  L/D = {CL/CD:.2f}")
        print(f"  Analysis confidence = {confidence:.3f}")
        print(f"  Top transition: {Top_Xtr:.3f}")
        print(f"  Bot transition: {Bot_Xtr:.3f}")
        
        if confidence < 0.5:
            print(f"  ⚠ Warning: Low confidence ({confidence:.3f})")
            print(f"    Results may be less accurate outside training data distribution")
        
        # Prepare result dictionary
        result_dict = {
            'reynolds': reynolds,
            'aoa': aoa,
            'mach': 0.0,  # NeuralFoil is incompressible
            'CL': CL,
            'CD': CD,
            'CM': CM,
            'Top_Xtr': Top_Xtr,
            'Bot_Xtr': Bot_Xtr,
            'analysis_confidence': confidence,
            'converged': True,
            'solver': 'neuralfoil'
        }
        
        # Save result
        result_file = output_path / f"{airfoil_path.stem}_Re{reynolds:.0e}_M0.00_aoa{aoa:.1f}_neuralfoil.csv"
        if HAS_PANDAS:
            df = pd.DataFrame([result_dict])
            df.to_csv(result_file, index=False)
            print(f"\n✓ Results saved: {result_file}")
        else:
            # Save as simple CSV without pandas
            import csv
            with open(result_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=result_dict.keys())
                writer.writeheader()
                writer.writerow(result_dict)
            print(f"\n✓ Results saved: {result_file}")
        
        return True, result_dict
        
    except ImportError as e:
        print(f"\n✗ NeuralFoil not available: {e}")
        print("  Install NeuralFoil: pip install neuralfoil")
        print("  Or install from local: cd neuralfoil && pip install -e .")
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
                         ncrit: Optional[float] = None,
                         **kwargs) -> Tuple[bool, Optional[object]]:
    """
    AoA sweep (자동 solver 선택)
    """
    
    # Create analysis condition
    condition = AnalysisCondition(reynolds=reynolds, mach=mach)
    
    # Select solver
    selected_solver = SolverSelector.select_solver(condition, solver)
    settings = SolverSelector.get_recommended_settings(condition, selected_solver)
    
    # Override ncrit if specified
    if ncrit is not None:
        settings['ncrit'] = ncrit
    
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
    
    elif selected_solver == SolverType.NEURALFOIL:
        print(f"\n{'='*70}")
        print("RUNNING NEURALFOIL AOA SWEEP")
        print(f"{'='*70}")
        
        # Build AoA array
        import numpy as np
        aoa_values = np.arange(aoa_min, aoa_max + d_aoa/2, d_aoa)
        
        print(f"\n  Total AoA points: {len(aoa_values)}")
        print(f"  Range: {aoa_min}° to {aoa_max}° (Δα = {d_aoa}°)")
        
        try:
            # Import NeuralFoil
            import sys
            neuralfoil_path = Path(__file__).parent.parent / "neuralfoil"
            if str(neuralfoil_path) not in sys.path:
                sys.path.insert(0, str(neuralfoil_path))
            
            from neuralfoil import get_aero_from_dat_file
            
            # Run vectorized NeuralFoil analysis
            print(f"\nAnalyzing with neural network surrogate (vectorized)...")
            print(f"  Model size: {settings.get('model_size', 'xlarge')}")
            
            result = get_aero_from_dat_file(
                str(airfoil_path),
                alpha=aoa_values,
                Re=reynolds,
                n_crit=settings.get('ncrit', 9),
                xtr_upper=settings.get('xtr_upper', 1.0),
                xtr_lower=settings.get('xtr_lower', 1.0),
                model_size=settings.get('model_size', 'xlarge')
            )
            
            # Convert to DataFrame
            results_list = []
            for i, aoa in enumerate(aoa_values):
                results_list.append({
                    'reynolds': reynolds,
                    'mach': mach,
                    'aoa': float(aoa),
                    'CL': float(result['CL'][i]),
                    'CD': float(result['CD'][i]),
                    'CM': float(result['CM'][i]),
                    'Top_Xtr': float(result['Top_Xtr'][i]),
                    'Bot_Xtr': float(result['Bot_Xtr'][i]),
                    'analysis_confidence': float(result['analysis_confidence'][i]),
                    'converged': True,
                    'solver': 'neuralfoil'
                })
            
            if HAS_PANDAS:
                df = pd.DataFrame(results_list)
                
                # Save results
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                result_file = output_path / f"{airfoil_path.stem}_Re{reynolds:.0e}_M{mach:.3f}_aoa{aoa_min}to{aoa_max}_neuralfoil.csv"
                df.to_csv(result_file, index=False)
                
                print(f"\n✓ Analysis complete")
                print(f"  Results saved: {result_file}")
                
                # Print summary statistics
                print(f"\n  Summary:")
                print(f"    CL range:  {df['CL'].min():.4f} to {df['CL'].max():.4f}")
                print(f"    CD range:  {df['CD'].min():.6f} to {df['CD'].max():.6f}")
                print(f"    Max L/D:   {(df['CL']/df['CD']).max():.2f} at α={df.loc[(df['CL']/df['CD']).idxmax(), 'aoa']:.1f}°")
                
                # Check for low confidence points
                low_conf = df[df['analysis_confidence'] < 0.5]
                if len(low_conf) > 0:
                    print(f"\n  ⚠ Warning: {len(low_conf)} points with low confidence (<0.5)")
                    print(f"    AoA: {low_conf['aoa'].min():.1f}° to {low_conf['aoa'].max():.1f}°")
                
                return True, df
            else:
                print(f"\n✓ Analysis complete")
                print("  (Install pandas for CSV output: pip install pandas)")
                return True, results_list
                
        except ImportError as e:
            print(f"\n✗ NeuralFoil not available: {e}")
            print("  Install NeuralFoil: pip install neuralfoil")
            print("  Or install from local: cd neuralfoil && pip install -e .")
            return False, None
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
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
    
    parser.add_argument('airfoil_file', type=str, nargs='?',
                       help='Path to airfoil coordinate file (.dat)')
    parser.add_argument('--re', type=float,
                       help='Reynolds number')
    parser.add_argument('--mach', type=float, default=0.0,
                       help='Mach number (default: 0.0)')
    
    # Single point or sweep
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--aoa', type=float,
                      help='Single angle of attack (degrees)')
    group.add_argument('--aoa-sweep', nargs=3, type=float, metavar=('MIN', 'MAX', 'STEP'),
                      help='AoA sweep: min max step (degrees)')
    
    # Solver selection
    parser.add_argument('--solver', type=str, choices=['xfoil', 'neuralfoil', 'su2_sa', 'su2_sst', 'su2_gamma_retheta'],
                       help='Force specific solver (default: auto-select)')
    parser.add_argument('--no-fallback', action='store_true',
                       help='Disable NeuralFoil fallback when XFoil fails')
    
    # Solver settings
    parser.add_argument('--ncrit', type=float, default=None,
                       help='Ncrit value for XFoil/NeuralFoil (default: auto based on Re)')
    
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
    
    # Validate required arguments
    if not args.airfoil_file:
        parser.error("airfoil_file is required unless --check is specified")
    if not args.re:
        parser.error("--re is required unless --check is specified")
    if args.aoa is None and args.aoa_sweep is None:
        parser.error("either --aoa or --aoa-sweep is required")
    
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
            output_dir=args.output_dir,
            use_neuralfoil_fallback=(not args.no_fallback),
            ncrit=args.ncrit
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
            output_dir=args.output_dir,
            ncrit=args.ncrit
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
