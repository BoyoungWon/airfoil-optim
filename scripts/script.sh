#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (edit as needed)
# -----------------------------
AIRFOIL_DIR="airfoil_samples_1000"
PY_SCRIPT="aoa_sweep.py"

# Reynolds number used in XFOIL viscous mode
RE="1000000"

# Optional: Ncrit (XFOIL transition parameter)
NCRIT="9"

# AoA sweep settings to compute ONLY alpha=0
AOA_MIN="0"
AOA_MAX="0"
DAOA="1"

# Output folders / files
OUT_DIR="results/aoa0_batch"
SUMMARY_CSV="${OUT_DIR}/summary_aoa0_Re${RE}_Ncrit${NCRIT}.csv"

# -----------------------------
# Checks
# -----------------------------
command -v xfoil >/dev/null 2>&1 || { echo "ERROR: xfoil not found in PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }

[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: $PY_SCRIPT not found in current directory"; exit 1; }
[[ -d "$AIRFOIL_DIR" ]] || { echo "ERROR: directory not found: $AIRFOIL_DIR"; exit 1; }

mkdir -p "$OUT_DIR"

# Write summary header
echo "airfoil,Re,Ncrit,alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr,csv_path" > "$SUMMARY_CSV"

# -----------------------------
# Run batch
# -----------------------------
shopt -s nullglob
FILES=("$AIRFOIL_DIR"/*.dat)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No .dat files found in ${AIRFOIL_DIR}"
  exit 1
fi

echo "Found ${#FILES[@]} airfoil files in ${AIRFOIL_DIR}"
echo "Writing summary to: ${SUMMARY_CSV}"
echo

FAIL_LOG="${OUT_DIR}/failed_files.txt"
: > "$FAIL_LOG"

for f in "${FILES[@]}"; do
  base="$(basename "$f")"
  name="${base%.*}"
  echo "=== [$name] Running AoA=0 @ Re=${RE}, Ncrit=${NCRIT} ==="

  # Run the python script
  if ! python3 "$PY_SCRIPT" "$f" "$RE" "$AOA_MIN" "$AOA_MAX" "$DAOA" "$NCRIT" >/dev/null 2>&1; then
    echo "  -> FAIL (python/xfoil error)"
    echo "$f" >> "$FAIL_LOG"
    continue
  fi

  # The python script writes CSV next to the polar txt:
  # results/aoa_sweep/<airfoil>_Re..._aoa0to0.csv
  # We search for the newest matching CSV containing "_aoa0to0"
  csv_path="$(ls -t results/aoa_sweep/"${name}"_Re*_aoa0to0*.csv 2>/dev/null | head -n 1 || true)"

  if [[ -z "${csv_path}" || ! -f "${csv_path}" ]]; then
    echo "  -> FAIL (could not find output CSV)"
    echo "$f" >> "$FAIL_LOG"
    continue
  fi

  # Extract the alpha=0 line (tolerant to "0", "0.0", "-0.0", etc.)
  # Columns from your script: alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr
  row="$(awk -F',' 'NR==1{next} { if ($1+0==0) {print; exit} }' "$csv_path" || true)"

  if [[ -z "$row" ]]; then
    echo "  -> FAIL (no alpha=0 row in CSV)"
    echo "$f" >> "$FAIL_LOG"
    continue
  fi

  echo "${name},${RE},${NCRIT},${row},${csv_path}" >> "$SUMMARY_CSV"
  echo "  -> OK"
done

echo
echo "Done."
echo "Summary: ${SUMMARY_CSV}"

if [[ -s "$FAIL_LOG" ]]; then
  echo "Some files failed. See: ${FAIL_LOG}"
else
  rm -f "$FAIL_LOG"
  echo "All files succeeded."
fi
