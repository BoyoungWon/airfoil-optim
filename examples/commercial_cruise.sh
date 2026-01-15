#!/bin/bash
# Commercial Aircraft Cruise - SU2 SA Analysis
# Boeing 737 / Airbus A320 class aircraft at cruise

echo "=================================================="
echo "Commercial Aircraft Cruise Analysis"
echo "=================================================="
echo "Conditions:"
echo "  - Reynolds: 1e7 (typical commercial cruise)"
echo "  - Mach: 0.78 (typical cruise speed)"
echo "  - Expected solver: SU2 SST (transonic)"
echo "=================================================="
echo ""

# RAE 2822 airfoil - benchmark transonic case
python scripts/unified_analysis.py input/airfoil/rae2822.dat \
    --re 6.5e6 \
    --mach 0.729 \
    --aoa 2.31 \
    --output-dir output/examples/commercial_cruise

echo ""
echo "⚠ Note: SU2 requires mesh generation"
echo "   Configuration file has been generated"
echo "   Generate mesh and run: SU2_CFD config.cfg"
echo ""
echo "Results will be in: output/examples/commercial_cruise"
