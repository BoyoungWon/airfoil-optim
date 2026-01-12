#!/usr/bin/env python3
"""
Airfoil Optimization Framework

통합 airfoil 최적화 실행 스크립트.
YAML 시나리오 파일을 읽어서 최적화를 수행합니다.

Usage:
    python scripts/optimize_airfoil.py --scenario scenarios/cruise_wing.yaml
    python scripts/optimize_airfoil.py --scenario scenarios/propeller.yaml --verbose
"""

import argparse
import sys
import yaml
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

# 최적화 관련
from scipy.optimize import minimize, differential_evolution
try:
    from skopt import gp_minimize
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False


class AirfoilOptimizer:
    """
    Airfoil 최적화 메인 클래스
    """
    
    def __init__(self, scenario_file: str):
        """
        Initialize optimizer with scenario file
        
        Parameters:
        -----------
        scenario_file : str
            Path to YAML scenario file
        """
        self.scenario_file = Path(scenario_file)
        self.config = self.load_scenario()
        
        # 출력 디렉토리 생성
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 초기화
        self.parametrization = None
        self.surrogate = None
        self.history = []
        
    def load_scenario(self) -> Dict:
        """Load and validate scenario file"""
        if not self.scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {self.scenario_file}")
        
        with open(self.scenario_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded scenario: {config['name']}")
        return config
    
    def setup_parametrization(self):
        """Setup shape parametrization based on config"""
        method = self.config['parametrization']['method']
        
        if method == 'naca':
            from optimize.parametrization import NACAParametrization
            self.parametrization = NACAParametrization(self.config['parametrization'])
        
        elif method == 'cst':
            from optimize.parametrization import CSTParametrization
            self.parametrization = CSTParametrization(self.config['parametrization'])
        
        elif method == 'ffd':
            from optimize.parametrization import FFDParametrization
            self.parametrization = FFDParametrization(self.config['parametrization'])
        
        else:
            raise ValueError(f"Unknown parametrization method: {method}")
        
        print(f"✓ Setup parametrization: {method}")
        print(f"  Number of parameters: {self.parametrization.n_params}")
    
    def setup_surrogate(self):
        """Setup surrogate model based on config"""
        method = self.config['surrogate']['method']
        
        if method == 'kriging':
            from optimize.surrogate import KrigingSurrogate
            self.surrogate = KrigingSurrogate(self.config['surrogate'])
        
        elif method == 'neural_network':
            from optimize.surrogate import NeuralNetworkSurrogate
            self.surrogate = NeuralNetworkSurrogate(self.config['surrogate'])
        
        elif method == 'polynomial':
            from optimize.surrogate import PolynomialSurrogate
            self.surrogate = PolynomialSurrogate(self.config['surrogate'])
        
        else:
            raise ValueError(f"Unknown surrogate method: {method}")
        
        print(f"✓ Setup surrogate model: {method}")
    
    def generate_training_data(self):
        """Generate training data for surrogate model"""
        n_samples = self.config['surrogate']['training_samples']
        
        print(f"\n📊 Generating {n_samples} training samples...")
        
        # Latin Hypercube Sampling
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.parametrization.n_params)
        samples = sampler.random(n=n_samples)
        
        # Scale to parameter bounds
        samples_scaled = self.parametrization.scale_samples(samples)
        
        # Evaluate with XFOIL
        X_train = []
        y_train = []
        
        for i, params in enumerate(samples_scaled):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_samples}")
            
            try:
                # Generate airfoil
                coords = self.parametrization.generate_airfoil(params)
                
                # Run XFOIL analysis
                results = self.evaluate_design_points(coords)
                
                if results is not None:
                    X_train.append(params)
                    y_train.append(results)
            
            except Exception as e:
                print(f"  Warning: Sample {i} failed - {e}")
                continue
        
        print(f"✓ Generated {len(X_train)} valid samples")
        
        return np.array(X_train), np.array(y_train)
    
    def evaluate_design_points(self, coords: np.ndarray) -> Dict:
        """
        Evaluate airfoil at all design points
        
        Returns dict with averaged/weighted results
        """
        from optimize.xfoil_interface import run_xfoil_analysis
        
        results_list = []
        weights = []
        
        for dp in self.config['design_points']:
            result = run_xfoil_analysis(
                coords,
                reynolds=dp['reynolds'],
                aoa=dp['aoa'],
                mach=dp.get('mach', 0.0)
            )
            
            if result is not None:
                results_list.append(result)
                weights.append(dp['weight'])
        
        if not results_list:
            return None
        
        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        
        combined = {}
        for key in results_list[0].keys():
            values = [r[key] for r in results_list]
            combined[key] = np.average(values, weights=weights)
        
        return combined
    
    def train_surrogate(self, X_train, y_train):
        """Train surrogate model"""
        print(f"\n🧠 Training surrogate model...")
        
        self.surrogate.train(X_train, y_train)
        
        # Validation
        val_split = self.config['surrogate']['validation_split']
        n_val = int(len(X_train) * val_split)
        
        if n_val > 0:
            X_val = X_train[-n_val:]
            y_val = y_train[-n_val:]
            X_train = X_train[:-n_val]
            y_train = y_train[:-n_val]
            
            score = self.surrogate.score(X_val, y_val)
            print(f"✓ Validation R² score: {score:.4f}")
    
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for optimization
        
        Evaluates using surrogate model
        """
        # Predict with surrogate
        prediction = self.surrogate.predict(params.reshape(1, -1))[0]
        
        # Calculate objective based on config
        obj_value = 0.0
        
        for obj_config in self.config['objectives']:
            metric = obj_config['metric']
            weight = obj_config['weight']
            obj_type = obj_config['type']
            
            if metric == 'CL/CD':
                value = prediction['CL'] / prediction['CD']
            elif metric == 'CL^1.5/CD':
                value = (prediction['CL'] ** 1.5) / prediction['CD']
            elif metric == 'CL':
                value = prediction['CL']
            elif metric == 'CD':
                value = prediction['CD']
            else:
                value = prediction.get(metric, 0.0)
            
            # Minimize or maximize
            if obj_type == 'maximize':
                obj_value += weight * value
            else:
                obj_value -= weight * value
        
        # For minimization algorithms, return negative for maximization
        self.history.append({
            'params': params.copy(),
            'objective': obj_value,
            'prediction': prediction.copy()
        })
        
        return -obj_value  # Negative because we minimize
    
    def check_constraints(self, params: np.ndarray) -> bool:
        """Check if design satisfies constraints"""
        prediction = self.surrogate.predict(params.reshape(1, -1))[0]
        
        # Check aerodynamic constraints
        if 'constraints' in self.config:
            constraints = self.config['constraints']
            
            if 'aerodynamic' in constraints:
                for constraint in constraints['aerodynamic']:
                    param = constraint['param']
                    value = prediction.get(param, 0.0)
                    
                    if 'min' in constraint and value < constraint['min']:
                        return False
                    if 'max' in constraint and value > constraint['max']:
                        return False
        
        return True
    
    def optimize(self):
        """Run optimization"""
        print(f"\n🎯 Starting optimization...")
        print(f"   Algorithm: {self.config['optimization']['algorithm']}")
        
        bounds = self.parametrization.get_bounds()
        x0 = self.parametrization.get_initial()
        
        algo = self.config['optimization']['algorithm']
        
        if algo == 'scipy':
            method = self.config['optimization'].get('method', 'SLSQP')
            result = minimize(
                self.objective_function,
                x0,
                method=method,
                bounds=bounds,
                options={
                    'maxiter': self.config['optimization']['max_iterations'],
                    'ftol': self.config['optimization']['convergence_tol']
                }
            )
        
        elif algo == 'genetic':
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=self.config['optimization']['max_generations'],
                popsize=self.config['optimization']['population_size']
            )
        
        elif algo == 'bayesian':
            if not HAS_SKOPT:
                raise ImportError("scikit-optimize required for Bayesian optimization")
            result = gp_minimize(
                self.objective_function,
                bounds,
                n_calls=self.config['optimization']['max_iterations'],
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        print(f"\n✓ Optimization complete!")
        print(f"   Final objective: {-result.fun:.6f}")
        print(f"   Iterations: {len(self.history)}")
        
        return result
    
    def save_results(self, result):
        """Save optimization results"""
        import json
        
        # Save optimal parameters
        optimal_params = result.x
        optimal_coords = self.parametrization.generate_airfoil(optimal_params)
        
        # Save airfoil
        output_file = self.output_dir / "optimal_airfoil.dat"
        with open(output_file, 'w') as f:
            f.write(f"Optimal {self.config['name']}\n")
            for x, y in optimal_coords:
                f.write(f"{x:12.8f}  {y:12.8f}\n")
        
        print(f"\n✓ Saved optimal airfoil: {output_file}")
        
        # Save parameters
        param_file = self.output_dir / "optimal_parameters.txt"
        np.savetxt(param_file, optimal_params)
        
        # Save history
        history_file = self.output_dir / "optimization_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"✓ Saved optimization history")
    
    def run(self):
        """Execute full optimization workflow"""
        print("=" * 60)
        print("AIRFOIL OPTIMIZATION")
        print("=" * 60)
        
        # Setup
        self.setup_parametrization()
        self.setup_surrogate()
        
        # Generate training data
        X_train, y_train = self.generate_training_data()
        
        # Train surrogate
        self.train_surrogate(X_train, y_train)
        
        # Optimize
        result = self.optimize()
        
        # Save results
        self.save_results(result)
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Airfoil Optimization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--scenario', type=str, required=True,
                       help='Path to scenario YAML file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = AirfoilOptimizer(args.scenario)
    optimizer.run()


if __name__ == "__main__":
    main()
