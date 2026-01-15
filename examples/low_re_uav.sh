#!/bin/bash
# Low Reynolds UAV Case - XFoil Analysis
# Typical small UAV or RC aircraft operating conditions

echo "=================================================="
echo "Low Reynolds UAV Analysis"
echo "=================================================="
echo "Conditions:"
echo "  - Reynolds: 2e5 (typical UAV cruise)"
echo "  - Mach: 0.1 (low speed)"
echo "  - Expected solver: XFoil"
echo "=================================================="
echo ""

# SD7037 airfoil - popular for small UAVs
python scripts/unified_analysis.py input/airfoil/sd7037.dat \
    --re 200000 \
    --mach 0.1 \
    --aoa-sweep 0 12 0.5 \
    --output-dir output/examples/uav_low_re

echo ""
echo "✓ UAV analysis complete"
echo "Results in: output/examples/uav_low_re"
