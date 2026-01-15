#!/usr/bin/env python3
"""
Solver Selection Demo

다양한 비행 조건에서의 자동 solver 선택을 시연합니다.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from solver_selector import (
    SolverType, AnalysisCondition, SolverSelector, get_solver_availability
)


def main():
    print("="*70)
    print("AUTOMATIC SOLVER SELECTION DEMO")
    print("="*70)
    print()
    
    # Check solver availability
    print("1. Checking installed solvers...")
    print("-"*70)
    avail = get_solver_availability()
    
    for solver_name, is_available in avail.items():
        status = "✓ Available" if is_available else "✗ Not installed"
        print(f"  {solver_name.upper():10s}: {status}")
    
    print()
    print("="*70)
    print("2. Testing Solver Selection for Different Flight Conditions")
    print("="*70)
    print()
    
    # Define test cases
    test_cases = [
        {
            'name': 'Small UAV / RC Aircraft',
            'condition': AnalysisCondition(reynolds=2e5, mach=0.1),
            'description': 'Low Re, low speed'
        },
        {
            'name': 'General Aviation',
            'condition': AnalysisCondition(reynolds=1e6, mach=0.25),
            'description': 'Medium Re, subsonic'
        },
        {
            'name': 'Regional Jet',
            'condition': AnalysisCondition(reynolds=5e6, mach=0.45),
            'description': 'High Re, subsonic'
        },
        {
            'name': 'Commercial Transport (Cruise)',
            'condition': AnalysisCondition(reynolds=1e7, mach=0.78),
            'description': 'Very high Re, transonic'
        },
        {
            'name': 'Business Jet (High Speed)',
            'condition': AnalysisCondition(reynolds=5e6, mach=0.85),
            'description': 'High Re, high transonic'
        },
        {
            'name': 'Edge Case: Very Low Re',
            'condition': AnalysisCondition(reynolds=5e4, mach=0.05),
            'description': 'Very low Re, very low speed'
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Case {i}: {case['name']}")
        print(f"{'='*70}")
        print(f"Description: {case['description']}")
        print(f"Reynolds:    {case['condition'].reynolds:.2e}")
        print(f"Mach:        {case['condition'].mach:.3f}")
        print()
        
        # Automatic selection
        solver = SolverSelector.select_solver(case['condition'])
        settings = SolverSelector.get_recommended_settings(case['condition'], solver)
        
        print(f"→ Selected Solver: {solver.value.upper()}")
        print(f"\nRecommended Settings:")
        for key, value in settings.items():
            print(f"  {key:20s}: {value}")
        
        # Show why this solver was chosen
        print(f"\nReasoning:")
        if solver == SolverType.XFOIL:
            if case['condition'].reynolds < 1e5:
                print("  ✓ Low Reynolds - XFoil with reduced Ncrit")
            elif case['condition'].reynolds < 1e6:
                print("  ✓ Medium Reynolds - XFoil optimal range")
            else:
                print("  ✓ XFoil acceptable but approaching limits")
        elif solver == SolverType.SU2_SA:
            if case['condition'].reynolds >= 1e6:
                print("  ✓ High Reynolds - RANS required")
            if 0.5 <= case['condition'].mach < 0.7:
                print("  ✓ Compressible flow - SA model adequate")
        elif solver == SolverType.SU2_SST:
            if case['condition'].mach >= 0.7:
                print("  ✓ Transonic flow - SST for shock-BL interaction")
                print("  ✓ Better handling of adverse pressure gradients")
    
    print()
    print("="*70)
    print("3. Manual Solver Override Example")
    print("="*70)
    print()
    
    # Example: Force XFoil for high Re (not recommended)
    condition = AnalysisCondition(reynolds=5e6, mach=0.3)
    print(f"Condition: Re={condition.reynolds:.2e}, Mach={condition.mach:.2f}")
    print()
    
    print("Auto-selection:")
    auto_solver = SolverSelector.select_solver(condition)
    print(f"  → {auto_solver.value.upper()}")
    print()
    
    print("Forcing XFoil (not recommended):")
    forced_solver = SolverSelector.select_solver(condition, SolverType.XFOIL)
    print(f"  → {forced_solver.value.upper()}")
    
    # Validate
    is_valid = SolverSelector.is_solver_valid(condition, SolverType.XFOIL)
    if not is_valid:
        print("  ⚠ Warning: XFoil may not be optimal for these conditions!")
    
    print()
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print()
    print("Key Takeaways:")
    print("  • Re < 1e6 and Mach < 0.5  → XFoil (fast, accurate)")
    print("  • Re ≥ 1e6 or Mach ≥ 0.5  → SU2 SA (compressible, high Re)")
    print("  • Mach ≥ 0.7               → SU2 SST (transonic shocks)")
    print()
    print("Usage:")
    print("  python scripts/unified_analysis.py AIRFOIL --re RE --mach MACH --aoa AOA")
    print()


if __name__ == "__main__":
    main()
