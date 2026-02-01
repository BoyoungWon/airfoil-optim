#!/usr/bin/env python3
"""
Batch Reynolds Sweep Analysis

이 스크립트는 input/airfoil 디렉토리의 모든 *.dat 파일에 대해
Reynolds number sweep 해석을 수행합니다.

Usage:
    python run_reynolds_sweep_batch.py [--aoa AOA] [--re-min MIN] [--re-max MAX] [--d-re STEP]
    
Example:
    python run_reynolds_sweep_batch.py --aoa 5.0 --re-min 100000 --re-max 5000000 --d-re 500000
"""

import argparse
import sys
from pathlib import Path
from reynolds_sweep import reynolds_sweep


def main():
    """
    Batch Reynolds sweep for all airfoils in input/airfoil directory
    """
    parser = argparse.ArgumentParser(
        description='Batch Reynolds sweep analysis for airfoils in input/airfoil directory'
    )
    parser.add_argument('--aoa', type=float, default=5.0,
                        help='Angle of attack in degrees (default: 5.0)')
    parser.add_argument('--re-min', type=float, default=100000,
                        help='Minimum Reynolds number (default: 1e5)')
    parser.add_argument('--re-max', type=float, default=5000000,
                        help='Maximum Reynolds number (default: 5e6)')
    parser.add_argument('--d-re', type=float, default=500000,
                        help='Reynolds number increment (default: 5e5)')
    parser.add_argument('--ncrit', type=float, default=9,
                        help='Ncrit parameter (default: 9)')
    parser.add_argument('--iter', type=int, default=100,
                        help='Maximum iterations (default: 100)')
    
    args = parser.parse_args()
    
    # Find input directory
    input_dir = Path("input/airfoil")
    
    if not input_dir.exists():
        print(f"✗ Error: Input directory not found: {input_dir}")
        print(f"  Please create the directory and add airfoil .dat files")
        sys.exit(1)
    
    # Find all .dat files
    dat_files = sorted(input_dir.glob("*.dat"))
    
    if not dat_files:
        print(f"✗ Error: No .dat files found in {input_dir}")
        print(f"  Please add airfoil coordinate files (.dat)")
        sys.exit(1)
    
    print("="*70)
    print("BATCH REYNOLDS SWEEP ANALYSIS")
    print("="*70)
    print(f"Input directory:  {input_dir.absolute()}")
    print(f"Found {len(dat_files)} airfoil file(s)")
    print(f"\nAnalysis parameters:")
    print(f"  Angle of attack:  {args.aoa}°")
    print(f"  Re range:         {args.re_min:.2e} to {args.re_max:.2e} (step: {args.d_re:.2e})")
    print(f"  Ncrit:            {args.ncrit}")
    print(f"  Max iterations:   {args.iter}")
    print("="*70)
    
    # Process each file
    results = []
    for i, dat_file in enumerate(dat_files, 1):
        print(f"\n[{i}/{len(dat_files)}] Processing: {dat_file.name}")
        print("-"*70)
        
        output_dir = f"output/analysis/reynolds_sweep/{dat_file.stem}"
        
        df, output_files = reynolds_sweep(
            str(dat_file),
            args.aoa,
            args.re_min,
            args.re_max,
            args.d_re,
            output_dir=output_dir,
            ncrit=args.ncrit,
            iter_limit=args.iter
        )
        
        if df is not None:
            results.append({
                'file': dat_file.name,
                'success': True,
                'points': len(df),
                'max_ld': (df['CL']/df['CD']).max() if len(df) > 0 else None
            })
        else:
            results.append({
                'file': dat_file.name,
                'success': False,
                'points': 0,
                'max_ld': None
            })
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH ANALYSIS SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"Total files:       {len(results)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {len(results) - successful}")
    
    if successful > 0:
        print("\nResults by airfoil:")
        for r in results:
            status = "✓" if r['success'] else "✗"
            if r['success']:
                print(f"  {status} {r['file']:30s}  Points: {r['points']:3d}  Max L/D: {r['max_ld']:.2f}")
            else:
                print(f"  {status} {r['file']:30s}  Failed")
    
    print("="*70)
    
    if successful == len(results):
        print("\n✓ All analyses completed successfully")
        sys.exit(0)
    elif successful > 0:
        print(f"\n⚠ {successful}/{len(results)} analyses completed successfully")
        sys.exit(0)
    else:
        print("\n✗ All analyses failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
