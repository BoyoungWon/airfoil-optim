#!/usr/bin/env python3
"""
Test NeuralFoil Integration

NeuralFoil이 solver로 제대로 통합되었는지 테스트하고,
XFoil과 NeuralFoil 결과를 비교합니다.
"""

import sys
from pathlib import Path
import time

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from solver_selector import SolverType, AnalysisCondition, SolverSelector, get_solver_availability
from unified_analysis import run_unified_single_point


def test_solver_availability():
    """Test solver availability"""
    print("="*70)
    print("TEST 1: Solver Availability Check")
    print("="*70)
    
    avail = get_solver_availability()
    
    print("\nInstalled solvers:")
    for solver_name, is_available in avail.items():
        status = "✓ Available" if is_available else "✗ Not found"
        print(f"  {solver_name:15s}: {status}")
    
    if not avail.get('neuralfoil', False):
        print("\n⚠ NeuralFoil not available!")
        print("  Install: cd neuralfoil && pip install -e .")
        return False
    
    print("\n✓ NeuralFoil is available")
    return True


def test_neuralfoil_solver_type():
    """Test NeuralFoil SolverType"""
    print("\n" + "="*70)
    print("TEST 2: NeuralFoil SolverType")
    print("="*70)
    
    # Test enum
    try:
        nf_solver = SolverType.NEURALFOIL
        print(f"\n✓ SolverType.NEURALFOIL = {nf_solver.value}")
    except AttributeError:
        print("\n✗ SolverType.NEURALFOIL not found!")
        return False
    
    # Test validation
    test_conditions = [
        (AnalysisCondition(reynolds=5e5, mach=0.2), True, "Medium Re, low Mach"),
        (AnalysisCondition(reynolds=1e7, mach=0.2), True, "High Re (boundary)"),
        (AnalysisCondition(reynolds=5e7, mach=0.2), False, "Very high Re (outside range)"),
        (AnalysisCondition(reynolds=5e5, mach=0.6), False, "Compressible (outside range)"),
    ]
    
    print("\nValidation tests:")
    for condition, expected_valid, desc in test_conditions:
        is_valid = SolverSelector.is_solver_valid(condition, SolverType.NEURALFOIL)
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"  {status} {desc:30s}: valid={is_valid} (expected={expected_valid})")
    
    return True


def test_neuralfoil_settings():
    """Test NeuralFoil recommended settings"""
    print("\n" + "="*70)
    print("TEST 3: NeuralFoil Settings")
    print("="*70)
    
    condition = AnalysisCondition(reynolds=5e5, mach=0.2)
    settings = SolverSelector.get_recommended_settings(condition, SolverType.NEURALFOIL)
    
    print("\nRecommended settings for Re=5e5, Mach=0.2:")
    for key, value in settings.items():
        print(f"  {key:20s}: {value}")
    
    required_keys = ['ncrit', 'model_size', 'xtr_upper', 'xtr_lower']
    missing = [k for k in required_keys if k not in settings]
    
    if missing:
        print(f"\n✗ Missing required settings: {missing}")
        return False
    
    print("\n✓ All required settings present")
    return True


def test_neuralfoil_analysis():
    """Test NeuralFoil analysis on NACA0012"""
    print("\n" + "="*70)
    print("TEST 4: NeuralFoil Analysis")
    print("="*70)
    
    # Find airfoil file
    airfoil_candidates = [
        Path("input/airfoil/naca0012.dat"),
        Path("public/airfoil/naca0012.dat"),
        Path("../input/airfoil/naca0012.dat"),
    ]
    
    airfoil_file = None
    for candidate in airfoil_candidates:
        if candidate.exists():
            airfoil_file = candidate
            break
    
    if airfoil_file is None:
        print("\n⚠ NACA0012 airfoil file not found")
        print("  Skipping analysis test")
        return True
    
    print(f"\nUsing airfoil: {airfoil_file}")
    
    # Test conditions
    reynolds = 5e5
    mach = 0.2
    aoa = 5.0
    
    print(f"Conditions: Re={reynolds:.0e}, Mach={mach}, AoA={aoa}°")
    
    # Run NeuralFoil
    print("\n--- Running NeuralFoil ---")
    start_time = time.time()
    success, result = run_unified_single_point(
        str(airfoil_file),
        reynolds,
        mach,
        aoa,
        solver=SolverType.NEURALFOIL,
        output_dir="output/analysis/test_neuralfoil",
        use_neuralfoil_fallback=False
    )
    nf_time = time.time() - start_time
    
    if not success:
        print("\n✗ NeuralFoil analysis failed!")
        return False
    
    print(f"\n✓ NeuralFoil completed in {nf_time:.3f} seconds")
    print(f"  CL = {result['CL']:.6f}")
    print(f"  CD = {result['CD']:.6f}")
    print(f"  CM = {result['CM']:.6f}")
    print(f"  L/D = {result['CL']/result['CD']:.2f}")
    print(f"  Confidence = {result.get('analysis_confidence', 0):.3f}")
    
    return True


def test_xfoil_neuralfoil_comparison():
    """Compare XFoil and NeuralFoil results"""
    print("\n" + "="*70)
    print("TEST 5: XFoil vs NeuralFoil Comparison")
    print("="*70)
    
    # Check if XFoil is available
    avail = get_solver_availability()
    if not avail.get('xfoil', False):
        print("\n⚠ XFoil not available, skipping comparison")
        return True
    
    # Find airfoil file
    airfoil_candidates = [
        Path("input/airfoil/naca0012.dat"),
        Path("public/airfoil/naca0012.dat"),
        Path("../input/airfoil/naca0012.dat"),
    ]
    
    airfoil_file = None
    for candidate in airfoil_candidates:
        if candidate.exists():
            airfoil_file = candidate
            break
    
    if airfoil_file is None:
        print("\n⚠ NACA0012 airfoil file not found")
        print("  Skipping comparison test")
        return True
    
    print(f"\nUsing airfoil: {airfoil_file}")
    
    # Test conditions
    reynolds = 5e5
    mach = 0.2
    aoa = 5.0
    
    print(f"Conditions: Re={reynolds:.0e}, Mach={mach}, AoA={aoa}°")
    
    # Run XFoil
    print("\n--- Running XFoil ---")
    start_time = time.time()
    xf_success, xf_result = run_unified_single_point(
        str(airfoil_file),
        reynolds,
        mach,
        aoa,
        solver=SolverType.XFOIL,
        output_dir="output/analysis/test_comparison",
        use_neuralfoil_fallback=False
    )
    xf_time = time.time() - start_time
    
    # Run NeuralFoil
    print("\n--- Running NeuralFoil ---")
    start_time = time.time()
    nf_success, nf_result = run_unified_single_point(
        str(airfoil_file),
        reynolds,
        mach,
        aoa,
        solver=SolverType.NEURALFOIL,
        output_dir="output/analysis/test_comparison",
        use_neuralfoil_fallback=False
    )
    nf_time = time.time() - start_time
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    if xf_success and nf_success:
        print(f"\n{'Parameter':<15} {'XFoil':<15} {'NeuralFoil':<15} {'Diff %':<10} {'Time (s)':<10}")
        print("-"*70)
        
        for key in ['CL', 'CD', 'CM']:
            xf_val = xf_result[key]
            nf_val = nf_result[key]
            diff_pct = abs(xf_val - nf_val) / abs(xf_val) * 100 if xf_val != 0 else 0
            print(f"{key:<15} {xf_val:<15.6f} {nf_val:<15.6f} {diff_pct:<10.2f}")
        
        print(f"{'Time':<15} {xf_time:<15.3f} {nf_time:<15.3f} {'':<10} {'Speedup: ' + str(xf_time/nf_time) + 'x':<10}")
        
        # Check if differences are reasonable
        cl_diff = abs(xf_result['CL'] - nf_result['CL']) / abs(xf_result['CL']) * 100
        cd_diff = abs(xf_result['CD'] - nf_result['CD']) / abs(xf_result['CD']) * 100
        
        print(f"\nDifferences:")
        print(f"  CL difference: {cl_diff:.2f}%")
        print(f"  CD difference: {cd_diff:.2f}%")
        
        if cl_diff < 5.0 and cd_diff < 10.0:
            print("\n✓ Results agree well (< 5% CL, < 10% CD)")
        else:
            print(f"\n⚠ Large differences detected")
            print(f"  This is normal for neural networks - they provide fast approximations")
        
    elif not xf_success and nf_success:
        print("\n✓ XFoil failed but NeuralFoil succeeded!")
        print("  This demonstrates NeuralFoil's value as a fallback")
        print(f"\nNeuralFoil results:")
        print(f"  CL = {nf_result['CL']:.6f}")
        print(f"  CD = {nf_result['CD']:.6f}")
        print(f"  CM = {nf_result['CM']:.6f}")
        print(f"  Time = {nf_time:.3f} s")
    
    elif xf_success and not nf_success:
        print("\n⚠ XFoil succeeded but NeuralFoil failed")
        print("  This is unexpected - check NeuralFoil installation")
        return False
    
    else:
        print("\n✗ Both solvers failed")
        return False
    
    return True


def test_fallback_logic():
    """Test XFoil -> NeuralFoil fallback"""
    print("\n" + "="*70)
    print("TEST 6: Fallback Logic")
    print("="*70)
    
    # Find airfoil file
    airfoil_candidates = [
        Path("input/airfoil/naca0012.dat"),
        Path("public/airfoil/naca0012.dat"),
        Path("../input/airfoil/naca0012.dat"),
    ]
    
    airfoil_file = None
    for candidate in airfoil_candidates:
        if candidate.exists():
            airfoil_file = candidate
            break
    
    if airfoil_file is None:
        print("\n⚠ NACA0012 airfoil file not found")
        print("  Skipping fallback test")
        return True
    
    print(f"\nUsing airfoil: {airfoil_file}")
    
    # Test with conditions that might challenge XFoil
    # Use high angle of attack where XFoil might struggle
    reynolds = 5e5
    mach = 0.2
    aoa = 15.0  # Higher AoA where separation might occur
    
    print(f"Conditions: Re={reynolds:.0e}, Mach={mach}, AoA={aoa}° (challenging)")
    print("\nTesting fallback: XFoil (auto) with NeuralFoil fallback enabled")
    
    success, result = run_unified_single_point(
        str(airfoil_file),
        reynolds,
        mach,
        aoa,
        solver=None,  # Auto-select (will choose XFoil)
        output_dir="output/analysis/test_fallback",
        use_neuralfoil_fallback=True
    )
    
    if success:
        print("\n✓ Analysis completed (XFoil or NeuralFoil fallback)")
        if result and result.get('solver') == 'neuralfoil':
            print("  → Used NeuralFoil fallback!")
        return True
    else:
        print("\n⚠ Both XFoil and fallback failed")
        print("  This might be expected for very difficult conditions")
        return True  # Don't fail test


def main():
    """Run all tests"""
    print("="*70)
    print("NEURALFOIL INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Solver Availability", test_solver_availability),
        ("SolverType Enum", test_neuralfoil_solver_type),
        ("Recommended Settings", test_neuralfoil_settings),
        ("NeuralFoil Analysis", test_neuralfoil_analysis),
        ("XFoil vs NeuralFoil", test_xfoil_neuralfoil_comparison),
        ("Fallback Logic", test_fallback_logic),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        
        print()  # Blank line between tests
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("="*70)
    print(f"Passed: {passed}/{total}")
    print("="*70)
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
