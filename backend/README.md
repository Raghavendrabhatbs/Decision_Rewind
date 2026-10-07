# Backend

This folder contains the FastAPI service for DECISION-REWIND.

## Purpose

- generate and persist synthetic data
- train and serve decision models
- compute provenance and counterfactual comparisons
- selectively rewind affected decisions
- verify actions deterministically
- expose a structured API for the UI

## Run

py -3.11 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
