# SkiSnow: Optimal Ski Resort Visit Prediction in the Alps

## Project Overview

**SkiSnow** is a predictive analytics project aimed at determining the best times to visit ski resorts across the Alps based on historical weather data and resort conditions. By analyzing factors such as snowfall, temperature, rainfall, and visibility, we provide actionable insights to enhance trip planning for ski enthusiasts and assist resorts in managing visitor expectations.

## Project Structure

- `data/`: Contains all raw and processed datasets.
  - `raw/`: Unmodified datasets directly downloaded from sources.
  - `processed/`: Cleaned and prepared datasets ready for analysis.
  - `external/`: Any external data used for the project.

- `notebooks/`: Jupyter notebooks for data collection, cleaning, EDA, modeling, and evaluation.

- `src/`: Source code for the project.
  - `data/`: Scripts to fetch and process data.
  - `features/`: Scripts for feature engineering.
  - `models/`: Scripts for training and evaluating models.
  - `visualization/`: Scripts for creating visualizations.

- `app/`: Streamlit application files for the web interface.

- `tests/`: Unit tests for the project modules.

- `requirements.txt`: Python package dependencies.

- `environment.yml`: Conda environment configuration.

- `setup.sh`: Shell script for setting up the project environment.

- `.gitignore`: Specifies intentionally untracked files to ignore.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Conda (optional but recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/SkiSnow.git
   cd SkiSnow
