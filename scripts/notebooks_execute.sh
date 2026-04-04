#!/usr/bin/env bash
# Headless notebook execution inside Docker (from repo root).
# Uses THESIS_FAST_NOTEBOOKS=1 so notebook 02 only runs the semi-annual schedule
# and notebook 05 fits a subset of assets (full runs: export THESIS_FAST_NOTEBOOKS=0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export THESIS_FAST_NOTEBOOKS="${THESIS_FAST_NOTEBOOKS:-1}"
mkdir -p results/_nbconvert
run_one() {
  docker compose run --rm -w /home/researcher/app \
    -e THESIS_FAST_NOTEBOOKS="$THESIS_FAST_NOTEBOOKS" \
    notebooks \
    jupyter nbconvert --to notebook --execute "$1" \
    --output-dir results/_nbconvert --output "$2" \
    --ExecutePreprocessor.timeout=7200
}
run_one notebooks/01_baseline_replication.ipynb 01_executed.ipynb
run_one notebooks/02_recalibration_sensitivity.ipynb 02_executed.ipynb
run_one notebooks/03_cluster_stability.ipynb 03_executed.ipynb
run_one notebooks/04_hypothesis_A_stress_and_stability.ipynb 04_executed.ipynb
run_one notebooks/05_hypothesis_B_calinski_harabasz_and_sample_size.ipynb 05_executed.ipynb
run_one notebooks/06_hypothesis_C_aggbond_silhouette_crises.ipynb 06_executed.ipynb
echo "Wrote results/_nbconvert/01–06_executed.ipynb"
