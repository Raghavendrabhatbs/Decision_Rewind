from __future__ import annotations

import json
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import APP_NAME
from backend.app.services.universal_log import (
    _decode_payload,
    install_application_log_capture,
    universal_log,
)

install_application_log_capture()

from backend.app.dataset.generator import generate_dataset
from backend.app.llm.provider import LLMProvider
from backend.app.ml.workbench_model import (
    TRAINING_EPOCHS,
    TRAINING_RECORD_COUNT,
    load_workbench_model,
    predict_workbench_decisions,
    save_training_dataset,
    train_model_version,
)
from backend.app.provenance.graph import build_provenance_graph
from backend.app.schemas.models import AIChatRequest
from backend.app.services.store import store
from backend.app.services.dataset_store import EDITABLE_FEATURES, dataset_store
from backend.app.services.llm_evidence import build_llm_evidence
from backend.app.provenance.graph import FEATURE_TO_DECISIONS


@asynccontextmanager
async def lifespan(_app: FastAPI):
    universal_log.record("application.started", "backend", {"app": APP_NAME})
    try:
        dataset_store.fail_interrupted_training_jobs()
        yield
    finally:
        universal_log.record("application.stopped", "backend", {"app": APP_NAME})


app = FastAPI(title=APP_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def capture_universal_api_events(request, call_next):
    if not request.url.path.startswith("/api/") or request.url.path == "/api/universal-log/client":
        return await call_next(request)

    started_at = time.perf_counter()
    request_body = await request.body()
    response = None
    try:
        response = await call_next(request)
        response_body = bytearray()
        async for chunk in response.body_iterator:
            response_body.extend(chunk)
        logged_response = _decode_payload(bytes(response_body))
        if request.url.path == "/api/ai/chat" and isinstance(logged_response, dict):
            logged_response.pop("evidence", None)
        universal_log.record(
            "api.request.completed",
            "backend",
            {
                "method": request.method,
                "path": request.url.path,
                "query": dict(request.query_params),
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "request": _decode_payload(request_body),
                "response": logged_response,
            },
        )
        return Response(
            content=bytes(response_body),
            status_code=response.status_code,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() != "content-length"
            },
            media_type=response.media_type,
            background=response.background,
        )
    except Exception as error:
        universal_log.record(
            "api.request.failed",
            "backend",
            {
                "method": request.method,
                "path": request.url.path,
                "query": dict(request.query_params),
                "status_code": response.status_code if response is not None else 500,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "request": _decode_payload(request_body),
                "error": str(error),
            },
        )
        raise


llm = LLMProvider()


@app.post("/api/universal-log/client", status_code=204)
def receive_frontend_log(payload: Dict[str, Any]) -> Response:
    event_type = payload.get("event_type")
    source = payload.get("source", "app")
    details = payload.get("details", {})
    if not isinstance(event_type, str) or not event_type.startswith("frontend."):
        raise HTTPException(status_code=422, detail="event_type must start with 'frontend.'.")
    if not isinstance(source, str) or not isinstance(details, dict):
        raise HTTPException(status_code=422, detail="source must be text and details must be an object.")
    if len(json.dumps(payload, default=str).encode("utf-8")) > 16_384:
        raise HTTPException(status_code=413, detail="Application log event exceeds the 16 KB limit.")
    universal_log.record(event_type, f"frontend.{source[:100]}", details)
    return Response(status_code=204)


def create_new_dataset(model: Dict[str, Any]) -> Dict[str, Any]:
    previous_seeds = {dataset["seed"] for dataset in dataset_store.list_datasets()}
    seed = secrets.randbelow(2**31)
    while seed in previous_seeds:
        seed = secrets.randbelow(2**31)
    created = datetime.now(timezone.utc)
    dataset_id = f"DS-{created.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"
    events = generate_dataset(size=200, seed=seed, include_demo_event=False, include_labels=False)
    try:
        dataset = dataset_store.create_dataset(dataset_id, seed, events, model)
        universal_log.record(
            "dataset.created",
            "dataset",
            {"dataset_id": dataset_id, "seed": seed, "record_count": len(events)},
        )
        universal_log.record_many(
            "security.event.generated",
            "dataset",
            [{"dataset_id": dataset_id, **event} for event in events],
        )
        return dataset
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def public_dataset_metadata(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dataset.items() if key != "model_path"}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": APP_NAME}


@app.get("/api/datasets")
def get_datasets() -> List[Dict[str, Any]]:
    return [public_dataset_metadata(dataset) for dataset in dataset_store.list_datasets()]


@app.get("/api/datasets/active")
def get_active_dataset() -> Dict[str, Any] | None:
    active = dataset_store.get_active_dataset()
    return public_dataset_metadata(active) if active is not None else None


@app.post("/api/datasets")
def generate_new_dataset() -> Dict[str, Any]:
    model = dataset_store.get_active_model()
    if model is None:
        raise HTTPException(status_code=409, detail="Train a global model before generating an experiment dataset.")
    return public_dataset_metadata(create_new_dataset(model))


@app.post("/api/datasets/{dataset_id}/train")
def train_dataset(dataset_id: str) -> Dict[str, Any]:
    dataset = dataset_store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset["training_status"] == "TRAINED":
        raise HTTPException(status_code=409, detail="This experiment has already been labeled.")
    try:
        trained_model = _load_pinned_model(dataset)
        events = dataset_store.get_training_events(dataset_id)
        if len(events) != 200:
            raise ValueError(f"Labeling requires exactly 200 experiment records; found {len(events)}.")
        predictions = {
            event["event_id"]: predict_workbench_decisions(event, trained_model)
            for event in events
        }
        universal_log.record(
            "model.predictions.generated",
            "workbench_model",
            {
                "dataset_id": dataset_id,
                "experiment_id": dataset["experiment_id"],
                "model_id": dataset["model_id"],
                "model_version": dataset["model_version"],
                "correlation_id": dataset_id,
                "predictions": predictions,
            },
        )
        labeled_dataset = dataset_store.complete_experiment_training(dataset_id, predictions)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=500, detail=f"Experiment labeling failed: {error}") from error
    return public_dataset_metadata(labeled_dataset)


def _run_training_job(job_id: str, training_dataset_id: str, seed: int, model_version: str) -> None:
    try:
        events = generate_dataset(size=TRAINING_RECORD_COUNT, seed=seed, include_demo_event=False)
        save_training_dataset(training_dataset_id, events)
        universal_log.record_many(
            "security.event.training_input",
            "workbench_model",
            [{"training_dataset_id": training_dataset_id, **event} for event in events],
        )
        metrics: Dict[str, Any] = {}

        def record_epoch(decision_id: str, epoch: int, stats: Dict[str, float]) -> None:
            metrics[decision_id] = {"epoch": epoch, **stats}
            dataset_store.update_training_job(job_id, decision_id, epoch, metrics)
            universal_log.record(
                "model.training.epoch",
                "workbench_model",
                {
                    "job_id": job_id,
                    "model_version": model_version,
                    "decision_id": decision_id,
                    "epoch": epoch,
                    "metrics": stats,
                },
            )

        model = train_model_version(training_dataset_id, seed, events, model_version, record_epoch)
        dataset_store.complete_training_job(job_id, model)
        universal_log.record("model.training.completed", "workbench_model", model)
    except Exception as error:
        dataset_store.fail_training_job(job_id, str(error))
        universal_log.record(
            "model.training.failed",
            "workbench_model",
            {"job_id": job_id, "model_version": model_version, "error": str(error)},
        )
        raise


@app.get("/api/models/status")
def get_model_status() -> Dict[str, Any]:
    return {
        "active_model": dataset_store.get_active_model(),
        "models": dataset_store.list_models(),
        "latest_training": dataset_store.get_latest_training_job(),
        "training_record_count": TRAINING_RECORD_COUNT,
        "epochs": TRAINING_EPOCHS,
    }


@app.post("/api/models/train")
def train_global_model(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if dataset_store.has_running_training_job():
        raise HTTPException(status_code=409, detail="A model training job is already running.")
    created = datetime.now(timezone.utc)
    job_id = f"TRAIN-{created.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"
    training_dataset_id = f"TRN-{created.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"
    seed = secrets.randbelow(2**31)
    model_version = dataset_store.next_model_version()
    try:
        dataset_store.create_training_job(
            job_id, model_version, training_dataset_id, seed, TRAINING_RECORD_COUNT, TRAINING_EPOCHS
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    background_tasks.add_task(_run_training_job, job_id, training_dataset_id, seed, model_version)
    return dataset_store.get_training_job(job_id) or {}


@app.get("/api/models/training/{job_id}")
def get_training_job(job_id: str) -> Dict[str, Any]:
    job = dataset_store.get_training_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@app.get("/api/datasets/{dataset_id}/events")
def get_dataset_events(
    dataset_id: str,
    search: str = "",
    sort_by: str = "event_id",
    descending: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    if not dataset_store.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(status_code=422, detail="Page must be positive and page_size must be between 1 and 100.")
    return dataset_store.list_events(dataset_id, search, sort_by, descending, page, page_size)


@app.get("/api/datasets/{dataset_id}/events/{event_id}")
def get_dataset_event(dataset_id: str, event_id: str) -> Dict[str, Any]:
    result = dataset_store.get_event(dataset_id, event_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Event not found in this dataset.")
    return result


def _validate_correction(feature: str, value: Any) -> None:
    if feature not in EDITABLE_FEATURES:
        raise HTTPException(status_code=422, detail=f"Feature {feature} cannot be corrected.")
    if feature == "asset_criticality" and value not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise HTTPException(status_code=422, detail="asset_criticality must be LOW, MEDIUM, HIGH, or CRITICAL.")
    if feature in {"failed_logins", "previous_alerts", "login_hour"}:
        maximum = 23 if feature == "login_hour" else (20 if feature == "failed_logins" else 10)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
            raise HTTPException(status_code=422, detail=f"{feature} must be an integer between 0 and {maximum}.")
    if feature == "geo_anomaly" and not isinstance(value, bool):
        raise HTTPException(status_code=422, detail="geo_anomaly must be a boolean.")
    if feature == "threat_intel_score" and (
        not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or value > 1
    ):
        raise HTTPException(status_code=422, detail="threat_intel_score must be between 0 and 1.")
    if feature == "source_ip" and not isinstance(value, str):
        raise HTTPException(status_code=422, detail="source_ip must be a string.")


def _validated_changes(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    changes = payload.get("changes")
    if not isinstance(changes, list) or not changes:
        raise HTTPException(status_code=422, detail="Provide at least one feature change.")
    normalized = []
    seen = set()
    for change in changes:
        if not isinstance(change, dict):
            raise HTTPException(status_code=422, detail="Each change must include feature and new_value.")
        feature = change.get("feature")
        value = change.get("new_value")
        if not isinstance(feature, str):
            raise HTTPException(status_code=422, detail="Feature name must be a string.")
        _validate_correction(feature, value)
        if feature in seen:
            raise HTTPException(status_code=422, detail=f"Feature {feature} was provided more than once.")
        seen.add(feature)
        normalized.append({"feature": feature, "new_value": value})
    return normalized


def _require_trained_dataset(dataset_id: str) -> Dict[str, Any]:
    dataset = dataset_store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset["training_status"] != "TRAINED" or not dataset.get("model_path"):
        raise HTTPException(status_code=409, detail="Train the active dataset before previewing or applying corrections.")
    return dataset


def _load_pinned_model(dataset: Dict[str, Any]) -> Dict[str, Any]:
    model_id = dataset.get("model_id")
    model_version = dataset.get("model_version")
    training_dataset_id = dataset.get("training_dataset_id")
    if not model_id or not model_version or not training_dataset_id:
        raise HTTPException(status_code=409, detail="The experiment is missing its pinned model metadata.")
    model = dataset_store.get_model(model_id)
    if (
        model is None
        or model["model_version"] != model_version
        or model["training_dataset_id"] != training_dataset_id
    ):
        raise HTTPException(status_code=409, detail="The model metadata pinned to this experiment is unavailable.")
    try:
        return load_workbench_model(model["artifact_path"], model["artifact_sha256"])
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail="The model artifact pinned to this experiment is unavailable or failed its integrity check.",
        ) from error


def _counterfactual_for_changes(
    dataset: Dict[str, Any],
    event: Dict[str, Any],
    changes: List[Dict[str, Any]],
) -> tuple[Dict[str, str], Dict[str, Any], List[str]]:
    corrected = dict(event)
    for change in changes:
        corrected[change["feature"]] = change["new_value"]
    counterfactual = predict_workbench_decisions(corrected, _load_pinned_model(dataset))
    changed_features = [change["feature"] for change in changes]
    return counterfactual, corrected, changed_features


def _decision_impact(
    decision_id: str,
    historical: str,
    current: str,
    counterfactual: str,
    changed_features: List[str],
    d2_changed: bool = False,
) -> Dict[str, Any]:
    relevant_features = [feature for feature in changed_features if decision_id in FEATURE_TO_DECISIONS.get(feature, [])]
    paths = [[feature, decision_id] for feature in relevant_features]
    if decision_id in {"D4", "D5"} and d2_changed:
        for feature in changed_features:
            if "D2" in FEATURE_TO_DECISIONS.get(feature, []) and feature not in relevant_features:
                relevant_features.append(feature)
                paths.append([feature, "D2", decision_id])
            elif "D2" in FEATURE_TO_DECISIONS.get(feature, []):
                paths.append([feature, "D2", decision_id])
    changed = counterfactual != historical
    requires_recovery = counterfactual != current
    return {
        "decision_id": decision_id,
        "historical_output": historical,
        "current_output": current,
        "counterfactual_output": counterfactual,
        "changed": changed,
        "requires_recovery": requires_recovery,
        "affected": requires_recovery and bool(relevant_features),
        "changed_features": relevant_features if changed else [],
        "dependency_paths": paths if changed else [],
    }


@app.post("/api/datasets/{dataset_id}/preview")
def preview_dataset_correction(dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = _require_trained_dataset(dataset_id)
    event_id = payload.get("event_id")
    changes = _validated_changes(payload)
    event_record = dataset_store.get_event(dataset_id, event_id)
    if event_record is None:
        raise HTTPException(status_code=404, detail="Event not found in this dataset.")
    current = event_record["current_state"]
    counterfactual, corrected, changed_features = _counterfactual_for_changes(dataset, current, changes)
    current_decisions = {row["decision_id"]: row for row in event_record["decisions"]}
    affected_decisions = []
    unaffected_decisions = []
    decision_impacts = []
    d2_changed = counterfactual["D2"] != current_decisions["D2"]["current_output"]
    for decision_id in ["D1", "D2", "D3", "D4", "D5"]:
        state = current_decisions[decision_id]
        result = _decision_impact(
            decision_id,
            state["historical_output"],
            state["current_output"],
            counterfactual[decision_id],
            changed_features,
            d2_changed,
        )
        decision_impacts.append(result)
        (affected_decisions if result["affected"] else unaffected_decisions).append(result)
    dataset_store.log_workflow_transition(
        dataset_id,
        event_id,
        "CORRECTION_PREVIEWED",
        {"changed_features": changed_features, "affected_decisions": [item["decision_id"] for item in affected_decisions]},
    )
    universal_log.record(
        "counterfactual.previewed",
        "dataset_workbench",
        {
            "dataset_id": dataset_id,
            "event_id": event_id,
            "correlation_id": f"{dataset_id}:{event_id}",
            "model_version": dataset["model_version"],
            "changes": [
                {"feature": feature, "old_value": current[feature], "new_value": corrected[feature]}
                for feature in changed_features
            ],
            "counterfactual_outputs": counterfactual,
            "affected_decisions": [item["decision_id"] for item in affected_decisions],
        },
    )
    return {
        "dataset_id": dataset_id,
        "event_id": event_id,
        "corrected_features": {feature: corrected[feature] for feature in changed_features},
        "changes": [
            {"feature": feature, "old_value": current[feature], "new_value": corrected[feature]}
            for feature in changed_features
        ],
        "changed_features": changed_features,
        "affected_decisions": affected_decisions,
        "unaffected_decisions": unaffected_decisions,
        "decision_impacts": decision_impacts,
        "historical_outputs": {key: row["historical_output"] for key, row in current_decisions.items()},
        "counterfactual_outputs": counterfactual,
        "graph": dataset_store.get_graph(dataset_id, changed_features),
    }


@app.post("/api/datasets/{dataset_id}/corrections")
def apply_dataset_correction(dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    event_id = payload.get("event_id")
    _require_trained_dataset(dataset_id)
    changes = _validated_changes(payload)
    try:
        result = dataset_store.create_correction(dataset_id, event_id, changes)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    universal_log.record(
        "workflow.correction.proposed",
        "dataset_workbench",
        {
            "dataset_id": dataset_id,
            "event_id": event_id,
            "correction_id": result["correction_id"],
            "correlation_id": f"{dataset_id}:{event_id}",
            "changes": result["changes"],
        },
    )
    return result


@app.post("/api/datasets/{dataset_id}/corrections/{correction_id}/cancel")
def cancel_dataset_correction(dataset_id: str, correction_id: str) -> Dict[str, Any]:
    correction = dataset_store.get_correction(correction_id)
    if correction is None or correction["dataset_id"] != dataset_id:
        raise HTTPException(status_code=404, detail="Correction not found in this dataset.")
    try:
        result = dataset_store.cancel_correction(correction_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return result or {}


@app.post("/api/datasets/{dataset_id}/preview/cancel")
def cancel_dataset_preview(dataset_id: str) -> Dict[str, str]:
    dataset = _require_trained_dataset(dataset_id)
    if dataset["workflow_status"] == "CORRECTION_PREVIEWED":
        dataset_store.log_workflow_transition(dataset_id, "", "READY_FOR_CORRECTION", {"preview_cancelled": True})
    return {"status": "READY_FOR_CORRECTION"}


@app.post("/api/datasets/{dataset_id}/rewind")
def rewind_dataset_decision(dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = _require_trained_dataset(dataset_id)
    correction = dataset_store.get_correction(payload.get("correction_id", ""))
    if correction is None or correction["dataset_id"] != dataset_id:
        raise HTTPException(status_code=404, detail="Correction not found in this dataset.")
    if correction["status"] != "PENDING":
        raise HTTPException(status_code=409, detail="This correction has already been rewound.")
    event_record = dataset_store.get_event(dataset_id, correction["event_id"])
    if event_record is None:
        raise HTTPException(status_code=404, detail="Event not found in this dataset.")
    current = event_record["current_state"]
    changes = [{"feature": item["feature"], "new_value": item["new_value"]} for item in correction["changes"]]
    changed = [item for item in changes if current[item["feature"]] != item["new_value"]]
    if not changed:
        raise HTTPException(status_code=422, detail="Proposed values must differ from the current feature values.")
    counterfactual, corrected, changed_features = _counterfactual_for_changes(dataset, current, changed)
    current_decisions = {row["decision_id"]: row for row in event_record["decisions"]}
    d2_changed = counterfactual["D2"] != current_decisions["D2"]["current_output"]
    decision_impacts = {
        decision_id: _decision_impact(
            decision_id,
            current_decisions[decision_id]["historical_output"],
            current_decisions[decision_id]["current_output"],
            counterfactual[decision_id],
            changed_features,
            d2_changed,
        )
        for decision_id in ["D1", "D2", "D3", "D4", "D5"]
    }
    affected = [
        decision_id
        for decision_id, impact in decision_impacts.items()
        if impact["affected"]
    ]
    reasons = []
    for decision_id, impact in decision_impacts.items():
        if impact["requires_recovery"] and not impact["affected"]:
            reasons.append(f"Decision {decision_id} changed without a dependency path from a corrected feature.")
    verification = {"verification": "VERIFIED" if not reasons else "REJECTED", "reasons": reasons}
    if verification["verification"] != "VERIFIED":
        raise HTTPException(status_code=409, detail=verification)
    try:
        result = dataset_store.complete_rewind(correction, counterfactual, affected, decision_impacts, verification)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    result.update({
        "dataset_id": dataset_id,
        "event_id": correction["event_id"],
        "affected_decisions": affected,
        "decision_impacts": decision_impacts,
        "counterfactual_outputs": counterfactual,
    })
    correlation_id = f"{dataset_id}:{correction['event_id']}"
    universal_log.record(
        "workflow.verification.completed",
        "dataset_workbench",
        {
            "dataset_id": dataset_id,
            "event_id": correction["event_id"],
            "correction_id": correction["correction_id"],
            "correlation_id": correlation_id,
            "verification": verification,
        },
    )
    universal_log.record(
        "workflow.rewind.completed",
        "dataset_workbench",
        {
            "dataset_id": dataset_id,
            "event_id": correction["event_id"],
            "correction_id": correction["correction_id"],
            "correlation_id": correlation_id,
            "model_version": dataset["model_version"],
            "affected_decisions": affected,
            "rewind_operations": result["rewind_operations"],
            "verification": verification["verification"],
        },
    )
    return result


@app.get("/api/datasets/{dataset_id}/graph")
def get_dataset_graph(dataset_id: str, feature: str | None = None) -> Dict[str, Any]:
    if not dataset_store.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset_store.get_graph(dataset_id, [feature] if feature else None)


@app.get("/api/datasets/{dataset_id}/audit")
def get_dataset_audit(dataset_id: str) -> List[Dict[str, Any]]:
    if not dataset_store.get_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset_store.list_audit(dataset_id, limit=None)


@app.get("/api/events")
def get_events() -> List[Dict[str, Any]]:
    raise HTTPException(status_code=410, detail="Legacy event API is disabled. Use /api/datasets and its event routes.")


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> Dict[str, Any]:
    raise HTTPException(status_code=410, detail="Legacy event API is disabled. Use /api/datasets/{dataset_id}/events.")


@app.post("/api/events/generate")
def generate_events(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise HTTPException(status_code=410, detail="Legacy event generation is disabled. Create an experiment with POST /api/datasets.")


@app.post("/api/corrections")
def create_correction(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise HTTPException(
        status_code=410,
        detail="Legacy correction API is disabled. Use the dataset correction preview and correction transaction routes.",
    )


@app.post("/api/rewind/analyze")
def analyze_rewind(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise HTTPException(
        status_code=410,
        detail="Legacy rule-based replay is disabled. Use POST /api/datasets/{dataset_id}/preview.",
    )


@app.post("/api/rewind/execute")
def execute_rewind(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise HTTPException(
        status_code=410,
        detail="Legacy recovery is disabled. Use the dataset correction transaction and POST /api/datasets/{dataset_id}/rewind.",
    )


@app.get("/api/decisions/{event_id}")
def get_decisions(event_id: str) -> List[Dict[str, Any]]:
    raise HTTPException(status_code=410, detail="Legacy decision API is disabled. Use /api/datasets/{dataset_id}/events/{event_id}.")


@app.get("/api/graph/{event_id}")
def get_graph(event_id: str) -> Dict[str, Any]:
    graph = build_provenance_graph()
    return {"nodes": [{"id": node, "kind": attrs.get("kind", "feature")} for node, attrs in graph.nodes(data=True)], "edges": [{"source": u, "target": v} for u, v in graph.edges()]}


@app.post("/api/ai/explain")
def ai_explain() -> Dict[str, Any]:
    raise HTTPException(
        status_code=410,
        detail="Legacy explanation payload is disabled. Use POST /api/ai/chat with dataset_id, event_id, and optional correction_id.",
    )


@app.post("/api/ai/chat")
def ai_chat(payload: AIChatRequest) -> Dict[str, Any]:
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Enter a question.")
    try:
        evidence = build_llm_evidence(
            question=payload.question,
            dataset_store=dataset_store,
            universal_log=universal_log,
            dataset_id=payload.dataset_id,
            experiment_id=payload.experiment_id,
            event_id=payload.event_id,
            correction_id=payload.correction_id,
            decision_id=payload.decision_id,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        response = llm.explain(payload.question, evidence)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    universal_log.record(
        "llm.model_output",
        "ai_chat",
        {"model": response["source"], "question": payload.question, "answer": response["answer"]},
    )
    return {"answer": response["answer"], "evidence": response["evidence"], "source": response["source"]}


@app.get("/api/ai/status")
def ai_status() -> Dict[str, Any]:
    return {
        "provider": llm.provider,
        "model": llm.model,
        "configured": bool(llm.api_key),
    }


@app.post("/api/experiments/run")
def run_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    raise HTTPException(
        status_code=410,
        detail="Legacy synthetic metrics are disabled. Use /api/datasets to create a pinned-model experiment.",
    )


@app.get("/api/audit")
def list_audit() -> List[Dict[str, Any]]:
    return store.list_audit()


@app.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    model = dataset_store.get_active_model()
    active = dataset_store.get_active_dataset()
    latest_verification = "NOT_AVAILABLE"
    if active:
        for record in dataset_store.list_audit(active["dataset_id"], limit=100):
            if record["operation"] == "VERIFIED":
                latest_verification = record["details"].get("verification", "VERIFIED")
                break
            if record["operation"] == "VERIFICATION_REJECTED":
                latest_verification = "REJECTED"
                break
    return {
        "datasets": {
            "training": model["training_record_count"] if model else None,
            "active_experiment": active["record_count"] if active else None,
        },
        "decision_count": 5,
        "training_model_version": model["model_version"] if model else None,
        "training_validation_metrics": model["validation_metrics"] if model else None,
        "verification_status": latest_verification,
    }
