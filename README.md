# Chess Opening ML CS 4641 Project

## Overview
This project explores machine learning techniques to analyze and predict outcomes of chess games based on openings.

This repository contains all necessary files and directories for the Chess Opening Machine Learning project, including preprocessing steps, model training, analysis, and visualization.

## Directories and Files

### `/archive/`
Contains earlier notebooks and midterm checkpoint files:
- `old_notebook.ipynb`: Initial notebook with early experimentation and mid-project model training.
- `midterm_checkpoint.ipynb`: Notebook used to present progress at the midterm checkpoint.

### `/data/`
Holds datasets for training, evaluation, and filtering:
- `games.csv`: Chess game data containing features like player ratings, moves, and outcomes.
- `high_elo_opening.csv`: Dataset specifically filtered for high ELO games, focusing on advanced openings.

### `/streamlit/`
Includes files for building an interactive web app to visualize the model results:
- `/data/`
    - `games.csv`: Chess game data containing features like player ratings, moves, and outcomes.
    - `high_elo_opening.csv`: Dataset specifically filtered for high ELO games, focusing on advanced openings.
- `main.py`: Streamlit app script for real-time interaction and visualization.
- `midtermreport.py`: Stores streamlit app submitted for midterm checkpoint.
- `proposal.py`: Stores streamlit app submitted for project proposal.
- `requirements.txt`: Dependency list for running the Streamlit application.

### `final_report.ipynb`
The main Jupyter notebook summarizing the project's progress, analysis, and final model evaluation.

### `testing.ipynb`
A notebook dedicated to testing various preprocessing methods and their impact on model performance.

### `README.md`
Provides an updated overview of the project, explaining the repository structure, files, and directories.
