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

## Universal event log

The API appends structured JSON Lines records to `data/universal_log.jsonl`.
It records API request and response payloads, including model results, and
training epoch metrics. Backend Python and Uvicorn logger records, frontend
click/change activity, frontend console warnings/errors, browser errors, and
application start/stop events are also recorded. Failed frontend API calls are
logged without their request bodies. AI chat and explanation requests include
the 12 most recent records, with large payloads clipped for context size.
Values under credential-like keys and common API/bearer key strings are
redacted before writing.

The log is local application data and is excluded from Git. The backend cannot
read unrelated VS Code or PowerShell terminal sessions or arbitrary terminal
stdout; only application logging and activity observed inside this application
are recorded.
