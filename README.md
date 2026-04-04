# Dynamic Asset Allocation with Asset-Specific Regime Forecasts

A Python replication of Shu, Yu & Mulvey (2024), *"Dynamic Asset Allocation with Asset-Specific Regime Forecasts"*, Journal of Portfolio Management.

This project implements a **Jump Model + XGBoost (JM-XGB)** framework that identifies bear and bull market regimes for 12 asset classes and uses those forecasts to build dynamic multi-asset portfolios.

---

## What this project does

The paper proposes a two-stage approach:

1. **Regime identification** — A Statistical Jump Model (JM) clusters daily asset returns into "bear" and "bull" regimes for each asset independently.
2. **Regime forecasting** — An XGBoost classifier predicts the next-day regime using historical return features and macroeconomic indicators.
3. **Portfolio construction** — Seven portfolio strategies (Equal Weight, Minimum Variance, Mean-Variance, and their regime-aware variants) are backtested from 2007 to 2022.

The replication uses freely available data:
- **Yahoo Finance** for asset prices (ETF proxies for the paper's Bloomberg indexes)
- **FRED** (Federal Reserve Economic Data) for macroeconomic features

---

## What you need to run this project

You only need **Docker Desktop** installed on your computer. No Python, no package managers, no coding experience required.

- **Mac**: [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
- **Windows**: [Download Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
- **Linux**: Follow the [Docker Engine install guide](https://docs.docker.com/engine/install/)

After installing, open Docker Desktop and wait until it shows "Docker is running" (the whale icon in the menu bar turns steady green on Mac).

---

## Quickstart

### Step 1 — Download this project

Open a terminal (Mac: `Terminal` app; Windows: `PowerShell` or `Command Prompt`) and run:

```bash
git clone https://github.com/eliabejr/shu_et_al_2024.git
cd shu_et_al_2024
```

If you do not have `git`, download the ZIP from the GitHub page and unzip it, then navigate to the folder in your terminal.

### Step 2 — Build the Docker image

This compiles all the required software into a self-contained image. It only needs to run once (or again after you update `requirements.txt`).

```bash
docker compose build
```

This takes 3–10 minutes depending on your internet speed. You will see a lot of text — that is normal.

### Step 3 — Launch Jupyter Lab

```bash
docker compose up notebooks
```

You will see output ending with lines like:

```
shu2024_notebooks  |     To access the server, open this URL in a browser:
shu2024_notebooks  |         http://127.0.0.1:8888/lab
```

Open your browser and go to **http://localhost:8888**. You will see the Jupyter Lab interface.

### Step 4 — Open a notebook

In the left panel, open the `notebooks/` folder. You will find three notebooks:

| Notebook | What it does |
|---|---|
| `01_baseline_replication.ipynb` | Full replication of Tables 4, 6, 7, 8, 9 and Figures 2–3 from the paper |
| `02_recalibration_sensitivity.ipynb` | Compares quarterly, semi-annual, and annual recalibration of the Jump Model |
| `03_cluster_stability.ipynb` | Measures how stable the regime labels are across consecutive estimation windows |

Click on `01_baseline_replication.ipynb` to start. Run all cells from top to bottom using the menu **Run → Run All Cells**.

> **Note on data download**: The first run downloads ~15 years of price and macro data from Yahoo Finance and FRED. This requires an internet connection and takes 2–5 minutes. Data is cached locally in `data/raw/` so subsequent runs are instant.

### Step 5 — Stop the server

Press `Ctrl+C` in the terminal where Docker is running, then:

```bash
docker compose down
```

---

## Running the full pipeline headlessly

If you want to reproduce all results without opening Jupyter, run the complete pipeline from the command line:

```bash
docker compose --profile pipeline up pipeline
```

Results (CSV files and figures) are saved to the `results/` folder on your machine.

---

## Running the test suite

To verify that your environment is correctly set up (no internet connection needed — all tests use synthetic data):

```bash
docker compose --profile tests up tests
```

You should see `12 passed` at the end.

---

## Project structure

```
shu_et_al_2024/
├── notebooks/                  # Interactive analysis notebooks
│   ├── 01_baseline_replication.ipynb
│   ├── 02_recalibration_sensitivity.ipynb
│   └── 03_cluster_stability.ipynb
├── src/
│   ├── config/settings.py      # All paper constants in one place
│   ├── data/                   # Data download and preprocessing
│   ├── features/               # Return and macro feature builders
│   ├── models/                 # Jump Model + XGBoost classifier
│   ├── portfolio/              # Optimizer and 7 portfolio strategies
│   ├── backtest/               # Simulation engine and performance metrics
│   └── utils/                  # Logging, plotting helpers
├── tests/                      # Smoke tests (synthetic data, no downloads)
├── pipeline.py                 # CLI entry point for headless runs
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Service orchestration
└── requirements.txt            # Python dependencies
```

---

## Differences from the paper

The paper uses proprietary data and software that are not publicly available. The following substitutions were made to enable full open-source replication:

| Paper | This replication |
|---|---|
| Bloomberg index data | Yahoo Finance ETF proxies (e.g., IVV for S&P 500) |
| Gurobi optimizer | CVXPY + OSQP (open-source, same formulation) |
| — | FRED via public graph CSV export for VIX and Treasury yields |

Results may differ slightly from the paper due to ETF tracking differences and minor data-vendor discrepancies. The algorithmic implementation is identical to the paper's description.

---

## Troubleshooting

**"Cannot connect to the Docker daemon"**
Docker Desktop is not running. Open the Docker Desktop application and wait for it to start.

**"Port 8888 is already in use"**
Another Jupyter session is already running on that port. Stop it first, or change the port in `docker-compose.yml` (e.g., `"8889:8888"`).

**Data download fails or is very slow**
Yahoo Finance and FRED rate-limit requests. Wait a few minutes and try again. Once cached in `data/raw/`, the download does not repeat.

**A notebook cell shows a red error**
Read the error message at the bottom of the cell. Most common causes: (1) run cells out of order — restart the kernel and run all cells from the top; (2) a network issue during data download — re-run the failing cell after waiting a moment.

---

## Citation

If you use this replication in your work, please cite the original paper:

> Shu, L., Yu, H., & Mulvey, J. M. (2024). Dynamic Asset Allocation with Asset-Specific Regime Forecasts. *The Journal of Portfolio Management*, 50(7), 74–97.

---

## License

This replication is released for academic and research use. The original paper's intellectual property belongs to its authors.
