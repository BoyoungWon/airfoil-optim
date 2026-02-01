#!/bin/bash
# Transonic Comparison - XFoil vs SU2
# Compare results at transition between XFoil and SU2 validity ranges

echo "=================================================="
echo "XFoil vs SU2 Comparison - Transonic Transition"
echo "=================================================="
echo ""

# Case 1: Medium Re, Low Mach (XFoil optimal)
echo "Case 1: Re=1e6, Mach=0.3 (XFoil optimal)"
echo "----------------------------------------"
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 1e6 \
    --mach 0.3 \
    --aoa 5.0 \
    --output-dir output/examples/comparison/case1_xfoil

echo ""
echo "✓ Case 1 complete"
echo ""

# Case 2: High Re, Moderate Mach (Transition zone)
echo "Case 2: Re=3e6, Mach=0.5 (Transition zone - SU2 recommended)"
echo "------------------------------------------------------------"
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 3e6 \
    --mach 0.5 \
    --aoa 5.0 \
    --output-dir output/examples/comparison/case2_su2

echo ""
echo "✓ Case 2 complete"
echo ""

# Case 3: High Re, High Mach (SU2 required)
echo "Case 3: Re=5e6, Mach=0.75 (Transonic - SU2 required)"
echo "----------------------------------------------------"
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 5e6 \
    --mach 0.75 \
    --aoa 2.5 \
    --output-dir output/examples/comparison/case3_transonic

echo ""
echo "✓ Case 3 complete"
echo ""

echo "=================================================="
echo "Comparison Summary"
echo "=================================================="
echo "Case 1: XFoil (Re=1e6, M=0.3) - Fast, accurate"
echo "Case 2: SU2 SA (Re=3e6, M=0.5) - Compressible effects"
echo "Case 3: SU2 SST (Re=5e6, M=0.75) - Shock waves"
echo ""
echo "Results in: output/examples/comparison/"
