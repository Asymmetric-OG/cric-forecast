<div align="center">

#  AI-Powered IPL Fantasy Score Predictor

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Transformers-F9AB00.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)

*A full-stack Machine Learning application that predicts T20 cricket fantasy points and evaluates player risk profiles using deep learning time-series forecasting (PatchTST).*

</div>

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features & Dashboard UI](#key-features--dashboard-ui)
3. [The Deep Learning Architecture (PatchTST)](#the-deep-learning-architecture-patchtst)
4. [Risk Profiling](#risk-profiling)
5. [Complete Tech Stack](#complete-tech-stack)
6. [Repository Structure](#repository-structure)
7. [Installation & Execution](#installation--execution)
8. [Training the Models](#training-the-models)
    
---

## Project Overview
Traditional fantasy cricket predictions rely on simple historical averages, which fail to capture the reality of a player's current form. This project completely reimagines player evaluation by treating cricket statistics as a complex **time-series problem**. 

By utilizing three separate **PatchTST Transformer neural networks**, the AI analyzes sliding windows of a player's last 10 matches to forecast their specific batting, bowling, and fielding metrics for the upcoming game. It then processes these raw outputs through a standard T20 Fantasy Points algorithm to generate the ultimate predicted XI.

---

## Key Features & Dashboard UI

The frontend is built on Streamlit with custom CSS theming, divided into two main analytical engines:

### 1. The Match Dashboard
Select two teams to instantly generate a predictive match simulation.
* **Best Predicted XI:** The top 11 players sorted by their AI-forecasted fantasy points.
* **Risk Profile Distribution:** A Plotly-powered visual breakdown of the match's safety vs. volatility.
* **Categorized Picks:** Automatically splits players into "Safe Picks" (Consistent anchors) and "High-Risk High-Reward" (Pinch hitters and wildcards).

### 2. Player Insights
Deep-dive analytics for individual players based on their specific ID and historical data.
* **Role Classification:** Automatically detects if a player is a Batsman, Bowler, All-Rounder, or Mixed based on their last 20 matches.
* **Performance Trend:** Visual line charts tracking their actual fantasy points over recent history.
* **Live Consistency Metrics:** Real-time display of their calculated Coefficient of Variation (CV).

---

## The Deep Learning Architecture (PatchTST)

The predictive engine (`src/inference.py`) does not use one giant model; it uses three highly specialized pipelines:

1. **The Batting Model:** Predicts `Runs`, `Balls Faced`, `Fours`, and `Sixes`.
2. **The Bowling Model:** Predicts `Runs Conceded`, `Balls Bowled`, `Dot Balls`, `Wickets`, `LBWs`, `Bowled`, and `Maiden Overs`.
3. **The Fielding Model:** Predicts `Catches`, `Caught & Bowled`, `Stumpings`, and `Run Outs`.

**The Sequence Logic (K=10):**
When a match is predicted, the script identifies all 22 players. It retrieves their last 10 historical matches (padding with zeros for rookies). This sequence is Z-score normalized using `data/scalers.json` and fed into the PyTorch models. The predicted metrics are reverse-scaled and passed through `fp_calculation.py` to calculate final fantasy scores.

---

## Risk Profiling

Because T20 cricket is incredibly volatile, predicting a static point value isn't enough. The dashboard calculates the **Coefficient of Variation (CV)** for every player to determine their reliability.

**The Formula:** `CV = Standard Deviation (σ) / Mean (μ)`

To account for the "spiky" nature of T20 scoring (where a player might score 80 points one day and 4 the next), the thresholds are uniquely calibrated for the T20 format:
* **Safe (CV < 0.75):** Highly consistent players (e.g., Anchors, Elite All-Rounders).
* **Moderate (CV < 1.0):** Standard top-order batsmen and strike bowlers.
* **High Risk (CV > 1.0):** Boom-or-bust players, lower-order sloggers, and part-time bowlers.
* *(Note: Rookies with less than 3 matches default to a 0.5 CV / Moderate rating).*

---

## Complete Tech Stack
| Component | Technologies Used |
| :--- | :--- |
| **Frontend UI** | Streamlit, Plotly, HTML/CSS |
| **Machine Learning** | PyTorch, Hugging Face Transformers (`PatchTSTConfig`) |
| **Data Processing** | Pandas, NumPy, JSON |

---

## Repository Structure

```text
cricpred/
│
├── app.py                  # Main Streamlit dashboard, caching, and UI routing
├── fp_calculation.py       # Standardized T20 Fantasy points rule engine
├── preprocess.py           # Aggregates raw JSONs into structured match-by-match CSVs
├── training.py             # Generates sliding windows, scales data, and trains PyTorch models
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores large model weights and generated data
│
├── src/                    # The Machine Learning Engine
│   └── inference.py        # Connects models to the dashboard, handles string-matching and Risk Math
│
├── data/                   # Generated data factory (Auto-generated, ignored by Git)
│   ├── t20_master/         # Folder containing master historical databases
│   ├── team_registry.json  # Maps team names to player IDs
│   └── player_registry.json# Maps player IDs to human-readable names
│
└── models/                 # PyTorch state dictionaries (Auto-generated, ignored by Git)
    ├── bat_model.pt
    ├── bowl_model.pt
    └── field_model.pt

```

---

## Installation & Execution

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/cricpred.git](https://github.com/YourUsername/cricpred.git)
cd cricpred
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run Preprocessing:**
```bash
python preprocess.py
```

**4. Launch the Dashboard:**
```bash
streamlit run app.py
```

---

## Training the Models

The pre-trained models can be generated locally. If you update your `data/` folder with new recent matches and want the AI to learn from them, you can completely retrain the neural networks from scratch.

Ensure your terminal is in the root `cricpred` folder and run:
```bash
python src/training.py
```

This script will:
1. Slice the aggregated CSV into 10-match sliding windows.
2. Recalculate the Z-score means and standard deviations, saving a fresh `scalers.json`.
3. Train the Batting, Bowling, and Fielding `PatchTST` models using the `AdamW` optimizer.
4. Save the new `.pt` weights into the `models/` folder.

Once finished, simply restart `app.py` to use the newly trained brains.
