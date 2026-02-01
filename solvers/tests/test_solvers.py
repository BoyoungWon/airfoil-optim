#!/usr/bin/env python3
"""
Unified Solvers Test

통합 solver 패키지 테스트
"""

import sys
from pathlib import Path

# Add solvers to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers import SolverType, get_available_solvers, select_solver
from solvers.unified import analyze, analyze_sweep, compare_solvers, print_comparison


def test_availability():
    """Test solver availability detection"""
    print("="*70)
    print("TEST: Solver Availability")
    print("="*70)
    
    avail = get_available_solvers()
    
    print("\nAvailable solvers:")
    for name, is_available in avail.items():
        status = "✓" if is_available else "✗"
        print(f"  {status} {name}")
    
    if not any(avail.values()):
        print("\n⚠ No solvers available!")
        return False
    
    return True


def test_solver_selection():
    """Test automatic solver selection"""
    print("\n" + "="*70)
    print("TEST: Solver Selection Logic")
    print("="*70)
    
    test_cases = [
        (5e5, 0.2, "Low Re, incompressible"),
        (1e6, 0.3, "Medium Re, incompressible"),
        (3e6, 0.75, "High Re, transonic"),
        (1e7, 0.85, "Very high Re, transonic"),
    ]
    
    print(f"\n{'Condition':<30} {'Selected Solver':<20}")
    print("-"*50)
    
    for re, mach, desc in test_cases:
        try:
            solver = select_solver(re, mach)
            print(f"{desc:<30} {solver.value:<20}")
        except Exception as e:
            print(f"{desc:<30} ERROR: {e}")
    
    return True


def test_neuralfoil_analysis():
    """Test NeuralFoil analysis"""
    print("\n" + "="*70)
    print("TEST: NeuralFoil Analysis")
    print("="*70)
    
    from solvers.neuralfoil_solver import NeuralFoilSolver
    
    if not NeuralFoilSolver.is_available():
        print("\n⚠ NeuralFoil not available, skipping")
        return True
    
    # Find airfoil file
    airfoil_candidates = [
        Path("input/airfoil/naca0012.dat"),
        Path("../input/airfoil/naca0012.dat"),
    ]
    
    airfoil_file = None
    for candidate in airfoil_candidates:
        if candidate.exists():
            airfoil_file = candidate
            break
    
    if airfoil_file is None:
        print("\n⚠ Airfoil file not found, skipping")
        return True
    
    solver = NeuralFoilSolver()
    result = solver.analyze(str(airfoil_file), reynolds=5e5, alpha=5.0)
    
    if result.get('converged'):
        print(f"\n✓ Analysis successful")
        print(f"  CL = {result['CL']:.6f}")
        print(f"  CD = {result['CD']:.6f}")
        print(f"  CM = {result['CM']:.6f}")
        print(f"  Confidence = {result.get('analysis_confidence', 'N/A')}")
        return True
    else:
        print(f"\n✗ Analysis failed: {result.get('error', 'Unknown')}")
        return False


def test_unified_analyze():
    """Test unified analyze function"""
    print("\n" + "="*70)
    print("TEST: Unified Analysis")
    print("="*70)
    
    airfoil_file = Path("input/airfoil/naca0012.dat")
    if not airfoil_file.exists():
        print("\n⚠ Airfoil file not found, skipping")
        return True
    
    result = analyze(str(airfoil_file), reynolds=5e5, alpha=5.0)
    
    if result.get('converged'):
        print(f"\n✓ Unified analysis successful")
        print(f"  Solver: {result.get('solver', 'unknown')}")
        print(f"  CL = {result['CL']:.6f}")
        print(f"  CD = {result['CD']:.6f}")
        return True
    else:
        print(f"\n✗ Analysis failed: {result.get('error', 'Unknown')}")
        return False


def test_analyze_sweep():
    """Test AoA sweep"""
    print("\n" + "="*70)
    print("TEST: AoA Sweep")
    print("="*70)
    
    airfoil_file = Path("input/airfoil/naca0012.dat")
    if not airfoil_file.exists():
        print("\n⚠ Airfoil file not found, skipping")
        return True
    
    import time
    start = time.time()
    
    results = analyze_sweep(
        str(airfoil_file),
        reynolds=5e5,
        alpha_range=(-5, 10, 2.5)
    )
    
    elapsed = time.time() - start
    
    converged_count = sum(1 for r in results if r.get('converged'))
    
    print(f"\n✓ Sweep completed in {elapsed:.3f}s")
    print(f"  Total points: {len(results)}")
    print(f"  Converged: {converged_count}/{len(results)}")
    
    if converged_count > 0:
        print(f"\n  Sample results:")
        print(f"  {'AoA':>8} {'CL':>10} {'CD':>10} {'L/D':>10}")
        print(f"  {'-'*40}")
        for r in results[:5]:
            if r.get('converged'):
                ld = r['CL'] / r['CD'] if r['CD'] != 0 else 0
                print(f"  {r['aoa']:>8.1f} {r['CL']:>10.4f} {r['CD']:>10.5f} {ld:>10.2f}")
    
    return converged_count > 0


def test_solver_comparison():
    """Test solver comparison"""
    print("\n" + "="*70)
    print("TEST: Solver Comparison")
    print("="*70)
    
    airfoil_file = Path("input/airfoil/naca0012.dat")
    if not airfoil_file.exists():
        print("\n⚠ Airfoil file not found, skipping")
        return True
    
    comparison = compare_solvers(str(airfoil_file), reynolds=5e5, alpha=5.0)
    
    if comparison:
        print_comparison(comparison)
        return True
    else:
        print("\n⚠ No solvers available for comparison")
        return True


def main():
    """Run all tests"""
    print("="*70)
    print("UNIFIED SOLVERS TEST SUITE")
    print("="*70)
    
    tests = [
        ("Solver Availability", test_availability),
        ("Solver Selection", test_solver_selection),
        ("NeuralFoil Analysis", test_neuralfoil_analysis),
        ("Unified Analysis", test_unified_analyze),
        ("AoA Sweep", test_analyze_sweep),
        ("Solver Comparison", test_solver_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("="*70)
    print(f"Passed: {passed}/{total}")
    print("="*70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
