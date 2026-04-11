#!/usr/bin/env bash
# Execute ablation notebooks 07–12 inside Docker (from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export THESIS_FAST_NOTEBOOKS="${THESIS_FAST_NOTEBOOKS:-0}"
TO="${NBEXEC_TIMEOUT_ABLATION:-43200}"
mkdir -p results/_nbconvert
run_one() {
  local nb="$1" out="$2"
  docker compose run --rm -w /home/researcher/app \
    -e THESIS_FAST_NOTEBOOKS="$THESIS_FAST_NOTEBOOKS" \
    notebooks \
    jupyter nbconvert --to notebook --execute "$nb" \
    --output-dir results/_nbconvert --output "$out" \
    --ExecutePreprocessor.timeout="$TO"
}
run_one notebooks/07_ablation_A_regime_identification.ipynb 07_executed.ipynb
run_one notebooks/08_ablation_B_forecasting.ipynb 08_executed.ipynb
run_one notebooks/09_ablation_C_allocation.ipynb 09_executed.ipynb
run_one notebooks/10_ablation_D_recalibration.ipynb 10_executed.ipynb
run_one notebooks/11_ablation_interactions.ipynb 11_executed.ipynb
run_one notebooks/12_ablation_variance_decomposition.ipynb 12_executed.ipynb
echo "Done: results/_nbconvert/07–12_executed.ipynb"
