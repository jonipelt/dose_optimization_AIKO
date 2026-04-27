# Dose Optimization Project

This project focuses on optimizing chemical dosing and initial pH based on raw water quality in order to achieve a target turbidity level after flotation.

The goal is to improve process efficiency, reduce chemical usage, and maintain consistent water quality.

## Note

This repository contains example code developed as part of the AIKO TAMK project. The project is co-financed by the EU. 

Link to the project: https://projects.tuni.fi/aiko/

- The original datasets are **not included**
- Trained model files are **not included**
- The code is provided for **demonstration and documentation purposes only**

The project is **not expected to run as-is** without project-specific data and models.

---

## Project Structure

- `src/` – Core source code  
  - preprocessing  
  - model training  
  - prediction & optimization  

- `data/` – Placeholder for data (no real production data included)

- `models/` – Storage for trained models (not included)

- `notebooks/` – Optional analysis and visualization (not included)

---

## Installation

```bash
python -m venv .venv              # Windows
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt   # Windows

python3 -m venv .venv             # Mac
source .venv/bin/activate         # Mac
pip install -r requirements.txt   # Mac
