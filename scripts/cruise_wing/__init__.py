"""
Cruise Wing Optimization Module

순항 익형 최적화를 위한 전용 모듈:
- NACA parametrization (3 variables)
- XFOIL solver
- Kriging surrogate
- SLSQP optimization

Workflow:
1. Database screening (NACA library scan)
2. Surrogate training (Kriging)
3. NACA optimization (SLSQP)
4. Validation with XFOIL

Application: 일반 항공기, 글라이더, UAV
Operating Condition (XFOIL Valid Range):
  - Re: 50k - 50M
  - Ma: 0 - 0.5 (with compressibility correction)
  - α: -5° - 15° (pre-stall region)
Design Objective: Maximize L/D at cruise
Complexity: ★☆☆☆☆ (가장 단순)
Timeline: 1-2일
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'CruiseWingOptimizer':
        from .workflow import CruiseWingOptimizer
        return CruiseWingOptimizer
    elif name == 'optimize_cruise_wing':
        from .workflow import optimize_cruise_wing
        return optimize_cruise_wing
    elif name == 'CruiseWingConfig':
        from .workflow import CruiseWingConfig
        return CruiseWingConfig
    elif name == 'DesignPoint':
        from .workflow import DesignPoint
        return DesignPoint
    elif name == 'NACADatabase':
        from .database import NACADatabase
        return NACADatabase
    elif name == 'AirfoilAnalyzer':
        from .analyzer import AirfoilAnalyzer
        return AirfoilAnalyzer
    elif name == 'CruiseWingKriging':
        from .kriging import CruiseWingKriging
        return CruiseWingKriging
    elif name == 'OptimizationVisualizer':
        from .visualizer import OptimizationVisualizer
        return OptimizationVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Main workflow
    'CruiseWingOptimizer',
    'optimize_cruise_wing',
    'CruiseWingConfig',
    'DesignPoint',
    
    # Components
    'NACADatabase', 
    'AirfoilAnalyzer',
    'CruiseWingKriging',
    'OptimizationVisualizer'
]
