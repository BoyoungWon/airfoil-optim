#!/usr/bin/env python3
"""
Scenario Validation Script

YAML 시나리오 파일의 유효성을 검증합니다.

Usage:
    python scripts/validate_scenario.py --scenario scenarios/cruise_wing.yaml
    python scripts/validate_scenario.py --all
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List
import sys


class ScenarioValidator:
    """YAML 시나리오 검증"""
    
    REQUIRED_FIELDS = [
        'name',
        'description',
        'category',
        'parametrization',
        'design_points',
        'objectives',
        'optimization',
        'surrogate',
        'output'
    ]
    
    VALID_PARAMETRIZATIONS = ['naca', 'cst', 'ffd']
    VALID_SURROGATE_METHODS = ['kriging', 'neural_network', 'polynomial']
    VALID_ALGORITHMS = ['scipy', 'genetic', 'bayesian']
    
    def __init__(self, scenario_file: Path):
        self.scenario_file = scenario_file
        self.config = None
        self.errors = []
        self.warnings = []
    
    def load_scenario(self) -> bool:
        """Load YAML file"""
        try:
            with open(self.scenario_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            return True
        except Exception as e:
            self.errors.append(f"Failed to load YAML: {e}")
            return False
    
    def validate_structure(self):
        """Validate required fields"""
        for field in self.REQUIRED_FIELDS:
            if field not in self.config:
                self.errors.append(f"Missing required field: {field}")
    
    def validate_parametrization(self):
        """Validate parametrization section"""
        if 'parametrization' not in self.config:
            return
        
        param = self.config['parametrization']
        
        if 'method' not in param:
            self.errors.append("parametrization.method is required")
            return
        
        method = param['method']
        if method not in self.VALID_PARAMETRIZATIONS:
            self.errors.append(f"Invalid parametrization method: {method}")
        
        # Method-specific validation
        if method == 'naca':
            if param.get('n_params', 3) != 3:
                self.warnings.append("NACA parametrization typically uses 3 parameters")
        
        elif method == 'cst':
            n_upper = param.get('n_upper', 6)
            n_lower = param.get('n_lower', 6)
            if n_upper < 3 or n_lower < 3:
                self.warnings.append("CST with < 3 parameters may be too restrictive")
            if n_upper + n_lower > 30:
                self.warnings.append("CST with > 30 parameters may require more training data")
        
        elif method == 'ffd':
            nx = param.get('nx', 5)
            ny = param.get('ny', 3)
            n_params = nx * ny * 2
            if n_params < 15:
                self.warnings.append("FFD with < 15 parameters may be too restrictive")
            if n_params > 100:
                self.warnings.append("FFD with > 100 parameters requires large training dataset")
    
    def validate_design_points(self):
        """Validate design points"""
        if 'design_points' not in self.config:
            return
        
        design_points = self.config['design_points']
        
        if not isinstance(design_points, list) or len(design_points) == 0:
            self.errors.append("design_points must be a non-empty list")
            return
        
        total_weight = 0.0
        
        for i, dp in enumerate(design_points):
            # Required fields
            if 'reynolds' not in dp:
                self.errors.append(f"design_points[{i}]: reynolds is required")
            if 'aoa' not in dp:
                self.errors.append(f"design_points[{i}]: aoa is required")
            if 'weight' not in dp:
                self.errors.append(f"design_points[{i}]: weight is required")
            else:
                total_weight += dp['weight']
            
            # Value ranges
            if 'reynolds' in dp:
                Re = dp['reynolds']
                # Handle string scientific notation (YAML 1.2 compatibility)
                if isinstance(Re, str):
                    try:
                        Re = float(Re)
                    except (ValueError, TypeError):
                        self.errors.append(f"design_points[{i}]: reynolds must be a number, got '{Re}'")
                        Re = None
                
                if isinstance(Re, (int, float)):
                    if Re < 1e4 or Re > 1e7:
                        self.warnings.append(f"design_points[{i}]: unusual Reynolds number {Re}")
                elif Re is not None:
                    self.errors.append(f"design_points[{i}]: reynolds must be a number, got {type(Re).__name__}")
            
            if 'aoa' in dp:
                aoa = dp['aoa']
                # Handle string scientific notation (YAML 1.2 compatibility)
                if isinstance(aoa, str):
                    try:
                        aoa = float(aoa)
                    except (ValueError, TypeError):
                        self.errors.append(f"design_points[{i}]: aoa must be a number, got '{aoa}'")
                        aoa = None
                
                if isinstance(aoa, (int, float)):
                    if aoa < -10 or aoa > 20:
                        self.warnings.append(f"design_points[{i}]: unusual AoA {aoa}°")
                elif aoa is not None:
                    self.errors.append(f"design_points[{i}]: aoa must be a number, got {type(aoa).__name__}")
        
        if abs(total_weight - 1.0) > 0.01:
            self.warnings.append(f"design_points weights sum to {total_weight}, not 1.0")
    
    def validate_objectives(self):
        """Validate objectives"""
        if 'objectives' not in self.config:
            return
        
        objectives = self.config['objectives']
        
        if not isinstance(objectives, list) or len(objectives) == 0:
            self.errors.append("objectives must be a non-empty list")
            return
        
        for i, obj in enumerate(objectives):
            if 'metric' not in obj:
                self.errors.append(f"objectives[{i}]: metric is required")
            if 'type' not in obj:
                self.errors.append(f"objectives[{i}]: type is required")
            elif obj['type'] not in ['maximize', 'minimize']:
                self.errors.append(f"objectives[{i}]: type must be 'maximize' or 'minimize'")
            if 'weight' not in obj:
                self.warnings.append(f"objectives[{i}]: weight not specified, using 1.0")
    
    def validate_optimization(self):
        """Validate optimization settings"""
        if 'optimization' not in self.config:
            return
        
        opt = self.config['optimization']
        
        if 'algorithm' not in opt:
            self.errors.append("optimization.algorithm is required")
        elif opt['algorithm'] not in self.VALID_ALGORITHMS:
            self.errors.append(f"Invalid optimization algorithm: {opt['algorithm']}")
        
        # Algorithm-specific validation
        algo = opt.get('algorithm')
        
        if algo == 'scipy':
            if 'max_iterations' not in opt:
                self.warnings.append("optimization.max_iterations not specified")
        
        elif algo == 'genetic':
            if 'max_generations' not in opt:
                self.warnings.append("optimization.max_generations not specified")
            if 'population_size' not in opt:
                self.warnings.append("optimization.population_size not specified")
        
        elif algo == 'bayesian':
            if 'max_iterations' not in opt:
                self.warnings.append("optimization.max_iterations not specified")
    
    def validate_surrogate(self):
        """Validate surrogate model settings"""
        if 'surrogate' not in self.config:
            return
        
        surr = self.config['surrogate']
        
        if 'method' not in surr:
            self.errors.append("surrogate.method is required")
        elif surr['method'] not in self.VALID_SURROGATE_METHODS:
            self.errors.append(f"Invalid surrogate method: {surr['method']}")
        
        # Check compatibility with parametrization
        param_method = self.config.get('parametrization', {}).get('method')
        surr_method = surr.get('method')
        
        if param_method == 'naca' and surr_method == 'neural_network':
            self.warnings.append("Neural network may be overkill for 3-parameter NACA")
        
        if param_method == 'ffd' and surr_method == 'polynomial':
            self.warnings.append("Polynomial may be too simple for high-dimensional FFD")
        
        # Training samples
        if 'training_samples' not in surr:
            self.warnings.append("surrogate.training_samples not specified")
        else:
            n_samples = surr['training_samples']
            
            # Get parameter count
            param = self.config.get('parametrization', {})
            if param.get('method') == 'naca':
                n_params = 3
            elif param.get('method') == 'cst':
                n_params = param.get('n_upper', 6) + param.get('n_lower', 6) + 1
            elif param.get('method') == 'ffd':
                n_params = param.get('nx', 5) * param.get('ny', 3) * 2
            else:
                n_params = 10  # Default
            
            # Rule of thumb: 10-20 samples per parameter
            recommended_min = n_params * 10
            recommended_max = n_params * 20
            
            if n_samples < recommended_min:
                self.warnings.append(
                    f"training_samples ({n_samples}) may be too few for {n_params} parameters. "
                    f"Recommended: {recommended_min}-{recommended_max}"
                )
    
    def validate(self) -> bool:
        """Run all validations"""
        if not self.load_scenario():
            return False
        
        self.validate_structure()
        self.validate_parametrization()
        self.validate_design_points()
        self.validate_objectives()
        self.validate_optimization()
        self.validate_surrogate()
        
        return len(self.errors) == 0
    
    def print_report(self):
        """Print validation report"""
        print(f"\n{'=' * 60}")
        print(f"Validation Report: {self.scenario_file.name}")
        print(f"{'=' * 60}\n")
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("✅ Scenario is valid with no warnings")
            return
        
        if len(self.errors) > 0:
            print(f"❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
            print()
        
        if len(self.warnings) > 0:
            print(f"⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
            print()
        
        if len(self.errors) == 0:
            print("✅ Scenario is valid (with warnings)")
        else:
            print("❌ Scenario is invalid")


def validate_all_scenarios(scenarios_dir: Path):
    """Validate all scenarios in directory"""
    scenario_files = list(scenarios_dir.glob("*.yaml"))
    
    if len(scenario_files) == 0:
        print(f"No YAML files found in {scenarios_dir}")
        return
    
    print(f"\n{'=' * 60}")
    print(f"Validating {len(scenario_files)} scenarios")
    print(f"{'=' * 60}\n")
    
    results = {}
    
    for scenario_file in sorted(scenario_files):
        validator = ScenarioValidator(scenario_file)
        is_valid = validator.validate()
        results[scenario_file.name] = {
            'valid': is_valid,
            'errors': len(validator.errors),
            'warnings': len(validator.warnings)
        }
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}\n")
    
    for name, result in sorted(results.items()):
        status = "✅" if result['valid'] else "❌"
        warnings = f" ({result['warnings']} warnings)" if result['warnings'] > 0 else ""
        errors = f" ({result['errors']} errors)" if result['errors'] > 0 else ""
        print(f"{status} {name}{errors}{warnings}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate airfoil optimization scenario files"
    )
    
    parser.add_argument('--scenario', type=str,
                       help='Path to scenario YAML file')
    parser.add_argument('--all', action='store_true',
                       help='Validate all scenarios in scenarios/ directory')
    
    args = parser.parse_args()
    
    if args.all:
        scenarios_dir = Path(__file__).parent.parent / "scenarios"
        validate_all_scenarios(scenarios_dir)
    
    elif args.scenario:
        scenario_file = Path(args.scenario)
        validator = ScenarioValidator(scenario_file)
        is_valid = validator.validate()
        validator.print_report()
        
        sys.exit(0 if is_valid else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
