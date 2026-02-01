#!/usr/bin/env python3
"""
Solver Selection Module

Re 수와 Mach 수에 따라 적절한 CFD solver를 선택하는 로직을 제공합니다.

Solver Selection Criteria:
- XFoil: Re < 1e6 and Mach < 0.5 (incompressible, low Re)
- SU2 RANS: Re >= 1e6 or Mach >= 0.5 (high Re, compressible)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class SolverType(Enum):
    """CFD Solver 종류"""
    XFOIL = "xfoil"
    SU2_SA = "su2_sa"           # Spalart-Allmaras
    SU2_SST = "su2_sst"         # k-omega SST
    SU2_GAMMA_RETHETA = "su2_gamma_retheta"  # Gamma-Re-theta transition


@dataclass
class AnalysisCondition:
    """해석 조건"""
    reynolds: float
    mach: float = 0.0
    altitude: Optional[float] = None  # meters
    temperature: Optional[float] = None  # Kelvin
    
    def __post_init__(self):
        """Mach 수로부터 압축성 판단"""
        if self.mach == 0.0 and self.reynolds > 0:
            # Default to low Mach if not specified
            self.mach = 0.1


class SolverSelector:
    """
    Re 수와 Mach 수에 따라 적절한 solver를 선택
    
    Selection Logic:
    ----------------
    1. Low Reynolds (Re < 1e5):
       - XFoil with lower Ncrit (5-7.5) for laminar-turbulent transition
       
    2. Medium Reynolds (1e5 <= Re < 1e6):
       - XFoil (optimal range)
       - Ncrit = 7.5-9 for standard conditions
       
    3. High Reynolds (Re >= 1e6):
       - SU2 RANS solvers recommended
       - SA model for attached flows
       - SST model for separated flows
       
    4. Compressible flow (Mach >= 0.5):
       - SU2 RANS required (XFoil invalid)
       
    5. Transonic flow (Mach >= 0.7):
       - SU2 with proper shock capturing
    """
    
    # Threshold values
    RE_LOW = 1e5
    RE_HIGH = 1e6
    MACH_SUBSONIC = 0.5
    MACH_TRANSONIC = 0.7
    MACH_SUPERSONIC = 1.2
    
    @classmethod
    def select_solver(cls, condition: AnalysisCondition, 
                     user_preference: Optional[SolverType] = None,
                     allow_auto: bool = True) -> SolverType:
        """
        최적의 solver 선택
        
        Parameters:
        -----------
        condition : AnalysisCondition
            해석 조건 (Re, Mach 등)
        user_preference : Optional[SolverType]
            사용자가 지정한 solver (None이면 자동 선택)
        allow_auto : bool
            자동 선택 허용 여부
            
        Returns:
        --------
        SolverType : 선택된 solver
        """
        
        # 사용자가 명시적으로 solver를 지정한 경우
        if user_preference is not None:
            # Validate user choice
            if not cls.is_solver_valid(condition, user_preference):
                print(f"⚠ Warning: {user_preference.value} may not be optimal for:")
                print(f"  Re = {condition.reynolds:.2e}, Mach = {condition.mach:.3f}")
                if not allow_auto:
                    print(f"  Proceeding with user-specified solver anyway...")
                    return user_preference
                else:
                    print(f"  Switching to automatic selection...")
            else:
                return user_preference
        
        # Automatic selection
        return cls._auto_select(condition)
    
    @classmethod
    def _auto_select(cls, condition: AnalysisCondition) -> SolverType:
        """자동 solver 선택 로직"""
        
        re = condition.reynolds
        mach = condition.mach
        
        # Rule 1: High Mach (compressible/transonic) -> SU2 required
        if mach >= cls.MACH_SUBSONIC:
            if mach >= cls.MACH_TRANSONIC:
                # Transonic: SST recommended for shock-boundary layer interaction
                return SolverType.SU2_SST
            else:
                # Subsonic compressible: SA is sufficient
                return SolverType.SU2_SA
        
        # Rule 2: High Reynolds -> SU2 RANS
        if re >= cls.RE_HIGH:
            # For high Re, SA model is generally good for attached flows
            # User can override to SST if separation is expected
            return SolverType.SU2_SA
        
        # Rule 3: Medium to Low Reynolds -> XFoil
        if re >= cls.RE_LOW:
            # XFoil optimal range
            return SolverType.XFOIL
        else:
            # Low Reynolds, XFoil can handle but may need special settings
            return SolverType.XFOIL
    
    @classmethod
    def is_solver_valid(cls, condition: AnalysisCondition, 
                       solver: SolverType) -> bool:
        """
        주어진 조건에서 solver가 유효한지 확인
        
        Returns:
        --------
        bool : True if valid, False if potentially problematic
        """
        
        re = condition.reynolds
        mach = condition.mach
        
        if solver == SolverType.XFOIL:
            # XFoil limitations
            if mach >= cls.MACH_SUBSONIC:
                return False  # XFoil is incompressible
            if re > cls.RE_HIGH:
                return False  # XFoil may struggle at very high Re
            return True
        
        elif solver in [SolverType.SU2_SA, SolverType.SU2_SST, 
                        SolverType.SU2_GAMMA_RETHETA]:
            # SU2 can handle any Re and Mach in this context
            # But may be overkill for low Re
            if re < cls.RE_LOW and mach < 0.3:
                # SU2 may be unnecessarily expensive for low Re, low Mach
                return True  # Still valid, just not optimal
            return True
        
        return False
    
    @classmethod
    def get_recommended_settings(cls, condition: AnalysisCondition, 
                                 solver: SolverType) -> dict:
        """
        Solver와 조건에 따른 권장 설정 반환
        
        Returns:
        --------
        dict : Solver-specific settings
        """
        
        re = condition.reynolds
        mach = condition.mach
        
        if solver == SolverType.XFOIL:
            # XFoil settings
            if re < cls.RE_LOW:
                ncrit = 5.0  # Lower Ncrit for low Re (more laminar)
            elif re < 5e5:
                ncrit = 7.5
            else:
                ncrit = 9.0  # Standard Ncrit
            
            return {
                'ncrit': ncrit,
                'iter_limit': 100,
                'viscous': True,
                'compressible': False
            }
        
        elif solver == SolverType.SU2_SA:
            return {
                'turbulence_model': 'SA',
                'mach': mach,
                'reynolds': re,
                'cfl': 5.0 if mach < 0.7 else 1.0,
                'mg_levels': 3,
                'iter': 5000 if mach < 0.7 else 10000
            }
        
        elif solver == SolverType.SU2_SST:
            return {
                'turbulence_model': 'SST',
                'mach': mach,
                'reynolds': re,
                'cfl': 3.0 if mach < 0.7 else 0.5,
                'mg_levels': 3,
                'iter': 10000 if mach < 0.7 else 20000
            }
        
        elif solver == SolverType.SU2_GAMMA_RETHETA:
            return {
                'turbulence_model': 'SA',
                'transition_model': 'LM',  # Langtry-Menter
                'mach': mach,
                'reynolds': re,
                'cfl': 2.0,
                'mg_levels': 2,
                'iter': 15000
            }
        
        return {}
    
    @classmethod
    def print_selection_info(cls, condition: AnalysisCondition, 
                            solver: SolverType,
                            settings: dict):
        """선택된 solver 정보 출력"""
        
        print("="*70)
        print("SOLVER SELECTION")
        print("="*70)
        print(f"Analysis Conditions:")
        print(f"  Reynolds number:  {condition.reynolds:.2e}")
        print(f"  Mach number:      {condition.mach:.3f}")
        
        # Flow regime classification
        if condition.mach < cls.MACH_SUBSONIC:
            regime = "Incompressible"
        elif condition.mach < cls.MACH_TRANSONIC:
            regime = "Subsonic Compressible"
        elif condition.mach < cls.MACH_SUPERSONIC:
            regime = "Transonic"
        else:
            regime = "Supersonic"
        print(f"  Flow regime:      {regime}")
        
        print(f"\nSelected Solver:    {solver.value.upper()}")
        
        print(f"\nRecommended Settings:")
        for key, value in settings.items():
            print(f"  {key:20s}: {value}")
        
        print("="*70)


def get_solver_availability() -> dict:
    """
    시스템에 설치된 solver 확인
    
    Returns:
    --------
    dict : {'xfoil': bool, 'su2': bool}
    """
    import subprocess
    
    availability = {}
    
    # Check XFoil
    try:
        subprocess.run(['xfoil', '-h'], capture_output=True, timeout=5)
        availability['xfoil'] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        availability['xfoil'] = False
    
    # Check SU2
    try:
        result = subprocess.run(['SU2_CFD', '-h'], capture_output=True, timeout=5)
        availability['su2'] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        availability['su2'] = False
    
    return availability


if __name__ == "__main__":
    """Test solver selection logic"""
    
    print("Testing Solver Selection Logic\n")
    
    test_cases = [
        AnalysisCondition(reynolds=5e4, mach=0.1),
        AnalysisCondition(reynolds=5e5, mach=0.2),
        AnalysisCondition(reynolds=3e6, mach=0.3),
        AnalysisCondition(reynolds=1e7, mach=0.75),
        AnalysisCondition(reynolds=5e6, mach=0.85),
    ]
    
    for condition in test_cases:
        solver = SolverSelector.select_solver(condition)
        settings = SolverSelector.get_recommended_settings(condition, solver)
        SolverSelector.print_selection_info(condition, solver, settings)
        print()
    
    # Check availability
    print("\nSolver Availability:")
    avail = get_solver_availability()
    for solver_name, is_available in avail.items():
        status = "✓ Available" if is_available else "✗ Not found"
        print(f"  {solver_name:10s}: {status}")
