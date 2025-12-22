# NFL Stats Predictions

This project is an ongoing exploration into using **machine learning** to predict **NFL player statistics** based on historical performance and contextual factors.

## Overview

The goal of this project is to build data-driven models that can estimate weekly NFL player production, including:

- **Running backs**
  - Rushing yards
  - Receiving yards
  - Total yards
- **Wide receivers**
  - Receiving yards
- **Quarterbacks**
  - Passing and rushing statistics (in progress)

The models are trained using **weekly player statistics** and incorporate information about both **player usage/performance trends** and **opposing defenses**.

## Approach

- Weekly NFL data is collected from the **nflverse** ecosystem.
- Features are engineered using rolling averages of recent performance (e.g. last few games).
- Defensive matchup context is included by aggregating how teams perform against specific positions.
- **Linear regression (Ridge regression)** is currently used as a baseline machine-learning model.
- Model performance is evaluated week-by-week to better reflect real-world forecasting scenarios.

## Current Focus

- Running back projections (RB1/RB2 usage, rushing + receiving production)
- Incorporating snap percentage and usage trends
- Clean terminal output and reproducible workflows

## Future Work

- Expand wide receiver and quarterback models
- Experiment with more advanced models (e.g. regularized regression, tree-based models)
- Improve evaluation methods and feature selection
- Add visualization and deeper analysis tools

## Disclaimer

This project is for **learning and experimentation purposes**. It is not intended to provide betting or fantasy guarantees, but rather to explore how machine learning techniques can be applied to real sports data.

---

*Project is actively under development.*
