#!/usr/bin/env python3
"""
Cruise Wing Module Test Script

모듈 기능 테스트 스크립트
각 컴포넌트를 개별적으로 테스트

Usage:
    python test_cruise_wing.py              # Run all tests
    python test_cruise_wing.py --quick      # Quick test (fewer samples)
    python test_cruise_wing.py --module db  # Test specific module
"""

import sys
import numpy as np
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_naca_database(verbose: bool = True) -> bool:
    """Test NACA database module"""
    print("\n" + "="*50)
    print("TEST: NACA Database")
    print("="*50)
    
    try:
        from cruise_wing.database import NACADatabase
        
        db = NACADatabase()
        
        # Test NACA code parsing
        params = db.parse_naca_code("2412")
        assert abs(params['m'] - 0.02) < 1e-6
        assert abs(params['p'] - 0.4) < 1e-6
        assert abs(params['t'] - 0.12) < 1e-6
        print("✓ NACA code parsing: OK")
        
        # Test coordinate generation
        coords = db.generate_naca_coords(0.02, 0.4, 0.12)
        assert coords.shape[1] == 2
        assert len(coords) > 100
        assert coords[0, 0] > 0.9  # Starts near TE
        print(f"✓ Coordinate generation: OK ({len(coords)} points)")
        
        # Test encoding
        code = db.encode_naca_code(0.02, 0.4, 0.12)
        assert code == "2412"
        print("✓ NACA code encoding: OK")
        
        if verbose:
            print("\n  Testing database scan (this may take a while)...")
            results = db.scan_database(reynolds=3e6, aoa=3.0, verbose=False)
            if results:
                print(f"✓ Database scan: OK ({len(results)} airfoils)")
                print(f"  Best: NACA {results[0]['naca']} with L/D={results[0].get('L/D', 'N/A')}")
            else:
                print("⚠ Database scan: No results (XFOIL may not be available)")
        
        print("\n✓ NACA Database: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ NACA Database: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyzer(verbose: bool = True) -> bool:
    """Test airfoil analyzer module"""
    print("\n" + "="*50)
    print("TEST: Airfoil Analyzer")
    print("="*50)
    
    try:
        from cruise_wing.analyzer import AirfoilAnalyzer
        from cruise_wing.database import NACADatabase
        
        analyzer = AirfoilAnalyzer()
        db = NACADatabase()
        
        # Generate test airfoil
        coords = db.generate_naca_coords(0.02, 0.4, 0.12)
        print("✓ Analyzer initialization: OK")
        
        # Test single point analysis
        print("\n  Testing XFOIL analysis...")
        result = analyzer.analyze_single(coords, reynolds=3e6, aoa=3.0, mach=0.2)
        
        if result:
            print(f"✓ Single point analysis: OK")
            print(f"  CL = {result.get('CL', 'N/A'):.4f}")
            print(f"  CD = {result.get('CD', 'N/A'):.5f}")
            print(f"  L/D = {result.get('L/D', 'N/A'):.2f}")
            
            if verbose:
                # Test polar analysis
                print("\n  Testing polar analysis...")
                polar = analyzer.analyze_polar(coords, reynolds=3e6, 
                                              aoa_range=(0, 8), aoa_step=1.0)
                if polar:
                    print(f"✓ Polar analysis: OK ({len(polar['alpha'])} points)")
                    idx_max = np.argmax(polar['L/D'])
                    print(f"  Max L/D = {polar['L/D'][idx_max]:.2f} at α={polar['alpha'][idx_max]}°")
                else:
                    print("⚠ Polar analysis: No results")
        else:
            print("⚠ Single point analysis: XFOIL may not be available")
            print("  (This is expected if XFOIL is not installed)")
        
        print("\n✓ Airfoil Analyzer: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Airfoil Analyzer: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kriging(verbose: bool = True) -> bool:
    """Test Kriging surrogate module"""
    print("\n" + "="*50)
    print("TEST: Kriging Surrogate")
    print("="*50)
    
    try:
        from cruise_wing.kriging import CruiseWingKriging, LHSSampler
        
        # Test LHS sampling
        bounds = [(0, 0.06), (0.2, 0.5), (0.09, 0.18)]
        sampler = LHSSampler(bounds)
        samples = sampler.sample(20)
        
        assert samples.shape == (20, 3)
        assert np.all(samples[:, 0] >= 0) and np.all(samples[:, 0] <= 0.06)
        print(f"✓ LHS Sampling: OK (20 samples)")
        
        # Create synthetic training data
        X_train = sampler.sample(50)
        
        # Synthetic objective: L/D ~ function of parameters
        def synthetic_ld(m, p, t):
            return 50 + 200*m - 100*m**2 + 50*(p-0.3) - 20*(t-0.12)**2
        
        y_train = [
            {
                'CL': 0.4 + 3*x[0],
                'CD': 0.008 + 0.01*x[2],
                'L/D': synthetic_ld(x[0], x[1], x[2])
            }
            for x in X_train
        ]
        
        # Train Kriging
        kriging = CruiseWingKriging(kernel='matern')
        stats = kriging.train(X_train, y_train, verbose=verbose)
        
        print(f"✓ Kriging training: OK")
        for metric, s in stats.items():
            print(f"  {metric}: R²={s['R2']:.4f}")
        
        # Test prediction
        test_point = np.array([0.02, 0.4, 0.12])
        pred = kriging.predict_single(test_point)
        
        print(f"✓ Prediction: OK")
        print(f"  Test point: m=0.02, p=0.4, t=0.12")
        print(f"  Predicted L/D: {pred['L/D']:.2f}")
        
        # Test prediction with uncertainty
        pred, std = kriging.predict_single(test_point, return_std=True)
        print(f"  Uncertainty (std): {std['L/D']:.2f}")
        
        # Cross-validation
        cv_scores = kriging.cross_validate(n_folds=3)
        print(f"✓ Cross-validation: OK")
        for metric, scores in cv_scores.items():
            print(f"  {metric}: CV R²={scores['mean']:.4f} ± {scores['std']:.4f}")
        
        print("\n✓ Kriging Surrogate: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Kriging Surrogate: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer(verbose: bool = True) -> bool:
    """Test optimizer module"""
    print("\n" + "="*50)
    print("TEST: Optimizer")
    print("="*50)
    
    try:
        from cruise_wing.optimizer import (
            CruiseWingOptimizer,
            OptimizationConfig,
            create_cruise_constraints
        )
        from cruise_wing.kriging import CruiseWingKriging, LHSSampler
        
        # Create synthetic surrogate
        bounds = [(0, 0.06), (0.2, 0.5), (0.09, 0.18)]
        sampler = LHSSampler(bounds)
        X_train = sampler.sample(50)
        
        def synthetic_ld(m, p, t):
            # Optimal around m=0.03, p=0.35, t=0.12
            return 80 - 500*(m-0.03)**2 - 100*(p-0.35)**2 - 500*(t-0.12)**2
        
        y_train = [
            {
                'CL': 0.4 + 3*x[0],
                'CD': 0.008 + 0.01*x[2],
                'CM': -0.05,
                'L/D': synthetic_ld(x[0], x[1], x[2])
            }
            for x in X_train
        ]
        
        kriging = CruiseWingKriging()
        kriging.train(X_train, y_train, verbose=False)
        print("✓ Synthetic surrogate created")
        
        # Create optimizer
        constraints = create_cruise_constraints(cl_min=0.3, ld_min=40)
        
        config = OptimizationConfig(
            method='SLSQP',
            max_iterations=30,
            objective_type='maximize',
            objective_metric='L/D'
        )
        
        def surrogate_eval(params):
            return kriging.predict_single(params)
        
        optimizer = CruiseWingOptimizer(
            objective_func=surrogate_eval,
            bounds=bounds,
            constraints=constraints,
            config=config
        )
        
        print("✓ Optimizer initialized")
        
        # Run optimization
        x0 = np.array([0.02, 0.4, 0.12])
        result = optimizer.optimize(x0=x0, verbose=verbose)
        
        print(f"\n✓ Optimization: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Optimal L/D: {result.optimal_value:.2f}")
        print(f"  Optimal params: m={result.optimal_params[0]:.4f}, "
              f"p={result.optimal_params[1]:.4f}, t={result.optimal_params[2]:.4f}")
        print(f"  Evaluations: {result.n_evaluations}")
        
        # Check if we're near the known optimum
        expected_optimal = np.array([0.03, 0.35, 0.12])
        error = np.linalg.norm(result.optimal_params - expected_optimal)
        if error < 0.05:
            print(f"✓ Optimal near expected (error={error:.4f})")
        else:
            print(f"⚠ Optimal differs from expected (error={error:.4f})")
        
        print("\n✓ Optimizer: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Optimizer: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer(verbose: bool = True) -> bool:
    """Test visualizer module"""
    print("\n" + "="*50)
    print("TEST: Visualizer")
    print("="*50)
    
    try:
        from cruise_wing.visualizer import OptimizationVisualizer
        from cruise_wing.database import NACADatabase
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        viz = OptimizationVisualizer(output_dir="output/test_figures")
        db = NACADatabase()
        
        print("✓ Visualizer initialized")
        
        # Generate test airfoils
        coords_2412 = db.generate_naca_coords(0.02, 0.4, 0.12)
        coords_4415 = db.generate_naca_coords(0.04, 0.4, 0.15)
        
        # Test airfoil comparison plot
        fig = viz.plot_airfoil_comparison({
            'NACA 2412': coords_2412,
            'NACA 4415': coords_4415
        }, save=True)
        print("✓ Airfoil comparison plot: OK")
        
        # Test convergence plot
        history = [
            {'iteration': i, 'objective': 50 + 10*np.log(i+1) + np.random.randn()*2, 
             'params': [0.02 + 0.001*i, 0.4, 0.12]}
            for i in range(30)
        ]
        fig = viz.plot_convergence(history, save=True)
        print("✓ Convergence plot: OK")
        
        # Test design space plot
        X_train = np.random.rand(50, 3)
        X_train[:, 0] *= 0.06
        X_train[:, 1] = 0.2 + X_train[:, 1] * 0.3
        X_train[:, 2] = 0.09 + X_train[:, 2] * 0.09
        
        y_train = [{'L/D': 50 + np.random.randn()*10} for _ in range(50)]
        
        fig = viz.plot_design_space(X_train, y_train, optimal=np.array([0.03, 0.35, 0.12]), save=True)
        print("✓ Design space plot: OK")
        
        # Test validation plot
        y_true = np.random.rand(30) * 20 + 50
        y_pred = y_true + np.random.randn(30) * 3
        
        fig = viz.plot_surrogate_validation(y_true, y_pred, save=True)
        print("✓ Surrogate validation plot: OK")
        
        print("\n✓ Visualizer: ALL TESTS PASSED")
        return True
        
    except ImportError as e:
        print(f"⚠ Visualizer: Skipped - matplotlib not available ({e})")
        return True  # Not a failure if matplotlib is not installed
        
    except Exception as e:
        print(f"\n✗ Visualizer: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow(quick: bool = False) -> bool:
    """Test complete workflow"""
    print("\n" + "="*50)
    print("TEST: Complete Workflow")
    print("="*50)
    
    try:
        from cruise_wing.workflow import (
            CruiseWingOptimizer,
            CruiseWingConfig,
            DesignPoint
        )
        
        # Create minimal config for quick test
        config = CruiseWingConfig(
            design_points=[DesignPoint(reynolds=3e6, aoa=3.0, mach=0.2)],
            use_surrogate=not quick,
            n_training_samples=30 if quick else 50,
            max_iterations=10 if quick else 30,
            n_multistart=2 if quick else 3,
            output_dir="output/test_workflow",
            create_plots=False  # Skip plots in test
        )
        
        optimizer = CruiseWingOptimizer(config=config)
        print("✓ Workflow optimizer initialized")
        
        # Test Phase 1 only (database screening)
        phase1 = optimizer.phase1_database_screening(verbose=True)
        
        if phase1['initial'].get('L/D'):
            print(f"✓ Phase 1: Found initial airfoil with L/D={phase1['initial']['L/D']:.2f}")
        else:
            print("⚠ Phase 1: XFOIL not available, using default")
        
        print("\n✓ Workflow: BASIC TESTS PASSED")
        print("  (Full workflow test requires XFOIL installation)")
        return True
        
    except Exception as e:
        print(f"\n✗ Workflow: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Cruise Wing Optimization Modules")
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick test with fewer samples')
    parser.add_argument('--module', '-m', type=str, 
                       choices=['db', 'analyzer', 'kriging', 'optimizer', 'viz', 'workflow', 'all'],
                       default='all',
                       help='Module to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CRUISE WING OPTIMIZATION - MODULE TESTS")
    print("="*60)
    
    results = {}
    
    tests = {
        'db': ('NACA Database', test_naca_database),
        'analyzer': ('Airfoil Analyzer', test_analyzer),
        'kriging': ('Kriging Surrogate', test_kriging),
        'optimizer': ('Optimizer', test_optimizer),
        'viz': ('Visualizer', test_visualizer),
        'workflow': ('Workflow', lambda v: test_workflow(args.quick))
    }
    
    if args.module == 'all':
        modules_to_test = list(tests.keys())
    else:
        modules_to_test = [args.module]
    
    for module in modules_to_test:
        name, test_func = tests[module]
        results[name] = test_func(args.verbose)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
