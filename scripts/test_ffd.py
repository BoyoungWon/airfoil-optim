#!/usr/bin/env python3
"""
FFD Airfoil Generator Test Script

Quick test script to verify FFD functionality and generate sample visualizations.
"""

import sys
from pathlib import Path
import subprocess


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ Success: {description}")
    else:
        print(f"\n❌ Failed: {description}")
        return False
    
    return True


def main():
    """Run FFD tests"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         FFD Airfoil Generator - Test Suite                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create test output directory
    test_dir = Path("test_ffd_output")
    test_dir.mkdir(exist_ok=True)
    print(f"📁 Test output directory: {test_dir.absolute()}\n")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Generate single FFD airfoil from NACA
    tests_total += 1
    if run_command(
        ["python", "scripts/ffd_airfoil.py", 
         "--naca", "0012",
         "--control-points", "5", "3",
         "--amplitude", "0.02",
         "-o", str(test_dir / "test_single_ffd.dat")],
        "Test 1: Single FFD airfoil from NACA 0012"
    ):
        tests_passed += 1
    
    # Test 2: Generate multiple samples
    tests_total += 1
    if run_command(
        ["python", "scripts/ffd_airfoil.py",
         "--naca", "2412",
         "--samples", "5",
         "--control-points", "4", "3",
         "--amplitude", "0.015",
         "--output-dir", str(test_dir / "samples"),
         "--seed", "42"],
        "Test 2: Generate 5 random samples from NACA 2412"
    ):
        tests_passed += 1
    
    # Test 3: Deform existing airfoil (if baseline exists)
    baseline_file = test_dir / "samples" / "NACA_2412_baseline.dat"
    if baseline_file.exists():
        tests_total += 1
        if run_command(
            ["python", "scripts/ffd_airfoil.py",
             "--input", str(baseline_file),
             "--control-points", "6", "3",
             "--amplitude", "0.01",
             "-o", str(test_dir / "test_from_existing.dat")],
            "Test 3: Deform existing airfoil file"
        ):
            tests_passed += 1
    
    # Test 4: Different control point configurations
    tests_total += 1
    if run_command(
        ["python", "scripts/ffd_airfoil.py",
         "--naca", "0012",
         "--control-points", "7", "3",
         "--amplitude", "0.025",
         "-o", str(test_dir / "test_7x3_control.dat")],
        "Test 4: FFD with 7×3 control points"
    ):
        tests_passed += 1
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"🎯 Test Summary")
    print(f"{'='*60}")
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ All tests passed!")
        print(f"\n📂 Check results in: {test_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(test_dir.rglob("*.dat")):
            size = file.stat().st_size
            print(f"  - {file.relative_to(test_dir)} ({size} bytes)")
    else:
        print(f"❌ {tests_total - tests_passed} test(s) failed")
        return 1
    
    print("\n" + "="*60)
    print("💡 Next steps:")
    print("  1. Visualize results: Use --plot flag with ffd_airfoil.py")
    print("  2. Run XFOIL analysis: Use aoa_sweep.py on generated airfoils")
    print("  3. Generate training data: Create large sample set for surrogate model")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
