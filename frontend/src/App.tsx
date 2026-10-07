import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, { Background, Controls, MarkerType, Position } from 'reactflow';
import 'reactflow/dist/style.css';

type EventRecord = {
  event_id: string;
  timestamp: string;
  user_id: string;
  source_ip: string;
  failed_logins: number;
  login_hour: number;
  geo_anomaly: boolean;
  asset_id: string;
  asset_type: string;
  asset_criticality: string;
  threat_intel_score: number;
  previous_alerts: number;
  decision_outputs: Record<string, string>;
  correction_status?: string;
  impact_status?: string;
};

type DatasetInfo = {
  dataset_id: string;
  seed: number;
  random_seed: number;
  created_at: string;
  generation_timestamp: string;
  record_count: number;
  is_active: number;
  training_status: 'NOT_TRAINED' | 'TRAINED';
  workflow_status: string;
  experiment_id?: string | null;
  model_id?: string | null;
  model_version?: string | null;
  training_record_count?: number | null;
  training_dataset_id?: string | null;
  trained_at?: string | null;
};

type ModelInfo = {
  model_id: string;
  model_version: string;
  training_dataset_id: string;
  training_record_count: number;
  epochs: number;
  best_epoch: Record<string, number>;
  trained_at: string;
  validation_metrics: Record<string, {
    training_loss: number;
    validation_loss: number;
    training_accuracy: number;
    validation_accuracy: number;
  }>;
  status: string;
};

type TrainingJob = {
  job_id: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  model_version: string;
  training_dataset_id: string;
  record_count: number;
  epochs: number;
  current_model?: string | null;
  current_epoch: number;
  metrics: Record<string, {
    epoch: number;
    training_loss: number;
    validation_loss: number;
    training_accuracy: number;
    validation_accuracy: number;
  }>;
  error?: string | null;
};

type ModelStatus = {
  active_model: ModelInfo | null;
  models: ModelInfo[];
  latest_training: TrainingJob | null;
  training_record_count: number;
  epochs: number;
};

type DecisionState = {
  decision_id: string;
  historical_output: string;
  current_output: string;
  counterfactual_output?: string | null;
  model_id: string;
  model_version: string;
  training_dataset_id: string;
};

type EventDetails = {
  original_state: EventRecord;
  current_state: EventRecord;
  decisions: DecisionState[];
  corrections: Array<{
    correction_id: string;
    feature_name: string;
    old_value: string;
    new_value: string;
    status: string;
    created_at: string;
  }>;
};

type PreviewDecision = {
  decision_id: string;
  historical_output: string;
  current_output: string;
  counterfactual_output: string;
  affected: boolean;
  changed: boolean;
  changed_features: string[];
  dependency_paths: string[][];
};

type ProvenanceGraphNode = { id: string; kind: 'feature' | 'decision' | 'output' };

type PreviewResult = {
  event_id: string;
  changed_features: string[];
  changes: Array<{ feature: string; old_value: unknown; new_value: unknown }>;
  affected_decisions: PreviewDecision[];
  unaffected_decisions: PreviewDecision[];
  decision_impacts: PreviewDecision[];
  historical_outputs: Record<string, string>;
  counterfactual_outputs: Record<string, string>;
  graph: { nodes: ProvenanceGraphNode[]; edges: Array<{ source: string; target: string }> };
};

type AuditRecord = {
  audit_id: string;
  event_id: string;
  operation: string;
  details: Record<string, unknown>;
  timestamp: string;
};

function auditLabel(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function AuditValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span>—</span>;
  if (typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return <span>{String(value)}</span>;
  }
  if (Array.isArray(value)) {
    return value.length
      ? <ul className="audit-value-list">{value.map((item, index) => <li key={index}><AuditValue value={item} /></li>)}</ul>
      : <span>None</span>;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value);
    return entries.length
      ? (
        <dl className="audit-detail-fields">
          {entries.map(([key, item]) => (
            <div key={key}>
              <dt>{auditLabel(key)}</dt>
              <dd><AuditValue value={item} /></dd>
            </div>
          ))}
        </dl>
      )
      : <span>None</span>;
  }
  return <span>{String(value)}</span>;
}

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const PAGE_SIZE = 20;
const DECISIONS = [
  { id: 'D1', label: 'Authentication' },
  { id: 'D2', label: 'Threat Severity' },
  { id: 'D3', label: 'Asset Protection' },
  { id: 'D4', label: 'Incident Escalation' },
  { id: 'D5', label: 'Response Action' },
];
const FEATURE_COLUMNS: Array<{ key: keyof EventRecord; label: string }> = [
  { key: 'event_id', label: 'Event ID' },
  { key: 'timestamp', label: 'Timestamp' },
  { key: 'user_id', label: 'User' },
  { key: 'source_ip', label: 'Source IP' },
  { key: 'failed_logins', label: 'Failed logins' },
  { key: 'login_hour', label: 'Login hour' },
  { key: 'geo_anomaly', label: 'Geo anomaly' },
  { key: 'asset_id', label: 'Asset ID' },
  { key: 'asset_type', label: 'Asset type' },
  { key: 'asset_criticality', label: 'Criticality' },
  { key: 'threat_intel_score', label: 'Threat score' },
  { key: 'previous_alerts', label: 'Previous alerts' },
];
const EDITABLE_FEATURES: Array<keyof EventRecord> = [
  'asset_criticality',
  'failed_logins',
  'login_hour',
  'geo_anomaly',
  'threat_intel_score',
  'previous_alerts',
  'source_ip',
];

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const request = { method: init?.method ?? 'GET', path: new URL(url, window.location.origin).pathname };
  try {
    response = await fetch(url, init);
  } catch (cause) {
    recordFrontendLog('frontend.api_request_failed', {
      ...request,
      error: cause instanceof Error ? cause.message : String(cause),
    });
    throw cause;
  }
  let body: { detail?: unknown };
  try {
    body = await response.json();
  } catch (cause) {
    recordFrontendLog('frontend.api_response_invalid', {
      ...request,
      status: response.status,
      error: cause instanceof Error ? cause.message : String(cause),
    });
    throw cause;
  }
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    const message = detail || `Request failed (${response.status}).`;
    recordFrontendLog('frontend.api_request_failed', { ...request, status: response.status, error: message });
    throw new Error(message);
  }
  return body as T;
}

const LOGGING_FAILURE_PREFIX = 'Could not save application log';

function recordFrontendLog(eventType: string, details: Record<string, unknown>, source = 'app') {
  void fetch(`${API_BASE}/api/universal-log/client`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event_type: eventType, source, details }),
    keepalive: true,
  }).then((response) => {
    if (!response.ok) console.error(`${LOGGING_FAILURE_PREFIX} (${response.status}).`);
  }).catch((cause: unknown) => {
    console.error(`${LOGGING_FAILURE_PREFIX}: could not reach the endpoint.`, cause);
  });
}

function formatConsoleArgument(value: unknown): string {
  if (value instanceof Error) return value.stack ?? value.message;
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function displayValue(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'YES' : 'NO';
  if (value === null || value === undefined) return '—';
  return String(value);
}

function formatDate(value: string): string {
  return new Date(value).toLocaleString();
}

export default function App() {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [latestTraining, setLatestTraining] = useState<TrainingJob | null>(null);
  const [rows, setRows] = useState<EventRecord[]>([]);
  const [totalRows, setTotalRows] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<keyof EventRecord>('event_id');
  const [descending, setDescending] = useState(false);
  const [details, setDetails] = useState<EventDetails | null>(null);
  const [proposedFeatures, setProposedFeatures] = useState<Record<string, unknown>>({});
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [correctionId, setCorrectionId] = useState('');
  const [rewindResult, setRewindResult] = useState<Record<string, unknown> | null>(null);
  const [graph, setGraph] = useState<PreviewResult['graph']>({ nodes: [], edges: [] });
  const [audit, setAudit] = useState<AuditRecord[]>([]);
  const [aiAnswer, setAiAnswer] = useState('');
  const [aiSource, setAiSource] = useState('');
  const [aiError, setAiError] = useState('');
  const [aiQuestion, setAiQuestion] = useState('Why would this correction affect these decisions?');
  const [visibleFeatures, setVisibleFeatures] = useState<string[]>(FEATURE_COLUMNS.map((column) => column.key));
  const [visibleDecisions, setVisibleDecisions] = useState<string[]>(DECISIONS.map((decision) => decision.id));
  const [busy, setBusy] = useState(false);
  const [training, setTraining] = useState(false);
  const [labeling, setLabeling] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const loadEvent = useCallback(async (datasetId: string, eventId: string) => {
    const result = await requestJson<EventDetails>(`${API_BASE}/api/datasets/${datasetId}/events/${eventId}`);
    setDetails(result);
    setProposedFeatures(Object.fromEntries(EDITABLE_FEATURES.map((name) => [name, result.current_state[name]])));
    setPreview(null);
    setGraph({ nodes: [], edges: [] });
    setCorrectionId('');
    setRewindResult(null);
    setAiAnswer('');
    setAiSource('');
    setAiError('');
  }, []);

  const loadAudit = useCallback(async (datasetId: string) => {
    const result = await requestJson<AuditRecord[]>(`${API_BASE}/api/datasets/${datasetId}/audit`);
    setAudit(result);
  }, []);

  const loadModelStatus = useCallback(async () => {
    const result = await requestJson<ModelStatus>(`${API_BASE}/api/models/status`);
    setModelStatus(result);
    setLatestTraining(result.latest_training);
    return result;
  }, []);

  const loadRows = useCallback(async (datasetInfo: DatasetInfo, nextPage: number, query: string, sort: keyof EventRecord, reverse: boolean) => {
    const params = new URLSearchParams({
      page: String(nextPage),
      page_size: String(PAGE_SIZE),
      search: query,
      sort_by: String(sort),
      descending: String(reverse),
    });
    const result = await requestJson<{ items: EventRecord[]; total: number; page: number }>(
      `${API_BASE}/api/datasets/${datasetInfo.dataset_id}/events?${params}`,
    );
    setRows(result.items);
    setTotalRows(result.total);
    if (result.items.length) {
      const chosenId = result.items.some((row) => row.event_id === details?.current_state.event_id)
        ? details?.current_state.event_id
        : result.items[0].event_id;
      if (chosenId) await loadEvent(datasetInfo.dataset_id, chosenId);
    } else {
      setDetails(null);
    }
  }, [details?.current_state.event_id, loadEvent]);

  const loadActiveDataset = useCallback(async () => {
    setBusy(true);
    setError('');
    try {
      const [active, status] = await Promise.all([
        requestJson<DatasetInfo | null>(`${API_BASE}/api/datasets/active`),
        loadModelStatus(),
      ]);
      setDataset(active);
      setPage(1);
      setGraph({ nodes: [], edges: [] });
      if (active) {
        await Promise.all([
          loadRows(active, 1, search, sortBy, descending),
          loadAudit(active.dataset_id),
        ]);
      } else {
        setRows([]);
        setTotalRows(0);
        setDetails(null);
        setAudit([]);
        setGraph({ nodes: [], edges: [] });
      }
      if (status.latest_training?.status === 'RUNNING') setTraining(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not load the active dataset.');
    } finally {
      setBusy(false);
    }
  }, [descending, loadAudit, loadModelStatus, loadRows, search, sortBy]);

  useEffect(() => {
    void loadActiveDataset();
  }, []);

  useEffect(() => {
    recordFrontendLog('frontend.application.started', { path: window.location.pathname });
    const describeTarget = (target: EventTarget | null) => {
      if (!(target instanceof HTMLElement)) return {};
      return {
        tag: target.tagName.toLowerCase(),
        id: target.id || undefined,
        name: target.getAttribute('name') || undefined,
        label: target.getAttribute('aria-label') || target.getAttribute('title') || undefined,
        text: target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement
          ? target.innerText.trim().slice(0, 100)
          : undefined,
      };
    };
    const onClick = (event: MouseEvent) => {
      recordFrontendLog('frontend.ui.click', {
        target: describeTarget(event.target),
        path: window.location.pathname,
      });
    };
    const onChange = (event: Event) => {
      recordFrontendLog('frontend.ui.change', {
        target: describeTarget(event.target),
        path: window.location.pathname,
      });
    };
    const onError = (event: ErrorEvent) => {
      recordFrontendLog('frontend.application_error', {
        message: event.message,
        filename: event.filename,
        line: event.lineno,
        column: event.colno,
      }, 'browser');
    };
    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      recordFrontendLog('frontend.unhandled_rejection', {
        reason: event.reason instanceof Error ? event.reason.message : String(event.reason),
      }, 'browser');
    };
    const onPageHide = () => {
      recordFrontendLog('frontend.application.stopped', { path: window.location.pathname });
    };
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    console.error = (...args: unknown[]) => {
      originalConsoleError.apply(console, args);
      if (String(args[0]).startsWith(LOGGING_FAILURE_PREFIX)) return;
      recordFrontendLog('frontend.console', {
        level: 'ERROR',
        message: args.map(formatConsoleArgument).join(' ').slice(0, 4000),
      }, 'browser');
    };
    console.warn = (...args: unknown[]) => {
      originalConsoleWarn.apply(console, args);
      recordFrontendLog('frontend.console', {
        level: 'WARNING',
        message: args.map(formatConsoleArgument).join(' ').slice(0, 4000),
      }, 'browser');
    };

    document.addEventListener('click', onClick, true);
    document.addEventListener('change', onChange, true);
    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);
    window.addEventListener('pagehide', onPageHide);
    return () => {
      document.removeEventListener('click', onClick, true);
      document.removeEventListener('change', onChange, true);
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
      window.removeEventListener('pagehide', onPageHide);
      console.error = originalConsoleError;
      console.warn = originalConsoleWarn;
    };
  }, []);

  useEffect(() => {
    if (error) recordFrontendLog('frontend.application_error', { area: 'workbench', message: error });
  }, [error]);

  useEffect(() => {
    if (aiError) recordFrontendLog('frontend.application_error', { area: 'ai_reasoning', message: aiError });
  }, [aiError]);

  useEffect(() => {
    if (!training || busy || latestTraining?.status !== 'RUNNING') return;
    let cancelled = false;
    const interval = window.setInterval(() => {
      void requestJson<TrainingJob>(`${API_BASE}/api/models/training/${latestTraining.job_id}`)
        .then((job) => {
          if (cancelled) return;
          setLatestTraining(job);
          if (job.status !== 'RUNNING') {
            setTraining(false);
            void loadModelStatus().catch((cause) => {
              setError(cause instanceof Error ? cause.message : 'Could not refresh model status.');
            });
            if (job.status === 'FAILED') setError(job.error || 'Model training failed.');
            else setMessage(`Training complete: SOC-MODEL-${job.model_version} trained on ${job.record_count.toLocaleString()} records for ${job.epochs} epochs.`);
          }
        })
        .catch((cause) => {
          if (!cancelled) {
            setTraining(false);
            setError(cause instanceof Error ? cause.message : 'Could not retrieve training progress.');
          }
        });
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [busy, latestTraining?.job_id, latestTraining?.status, loadModelStatus, training]);

  useEffect(() => {
    if (!dataset) return;
    let cancelled = false;
    const load = async () => {
      setBusy(true);
      try {
        const params = new URLSearchParams({
          page: String(page),
          page_size: String(PAGE_SIZE),
          search,
          sort_by: String(sortBy),
          descending: String(descending),
        });
        const result = await requestJson<{ items: EventRecord[]; total: number }>(
          `${API_BASE}/api/datasets/${dataset.dataset_id}/events?${params}`,
        );
        if (cancelled) return;
        setRows(result.items);
        setTotalRows(result.total);
        if (result.items.length) {
          const selectedStillVisible = result.items.some((row) => row.event_id === details?.current_state.event_id);
          if (!selectedStillVisible) await loadEvent(dataset.dataset_id, result.items[0].event_id);
        } else {
          setDetails(null);
        }
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : 'Could not load dataset records.');
      } finally {
        if (!cancelled) setBusy(false);
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [dataset?.dataset_id, descending, page, search, sortBy]);

  const pageCount = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));
  const selectedEvent = details?.current_state;
  const selectedDecision = (id: string) => details?.decisions.find((item) => item.decision_id === id);
  const changedFeatures = selectedEvent
    ? EDITABLE_FEATURES.filter((name) => proposedFeatures[name] !== selectedEvent[name])
    : [];
  const changedValues = changedFeatures.map((name) => ({ feature: name, new_value: proposedFeatures[name] }));
  const previewMatchesProposal = Boolean(
    preview
      && preview.event_id === selectedEvent?.event_id
      && preview.changes.length === changedValues.length
      && changedValues.every((change) => preview.changes.some(
        (item) => item.feature === change.feature && item.new_value === change.new_value,
      )),
  );
  const graphNodes = useMemo(() => {
    const featureNodes = graph.nodes
      .filter((node) => node.kind === 'feature')
      .sort((left, right) => FEATURE_COLUMNS.findIndex((column) => column.key === left.id) - FEATURE_COLUMNS.findIndex((column) => column.key === right.id));
    const decisionNodes = graph.nodes
      .filter((node) => node.kind === 'decision')
      .sort((left, right) => DECISIONS.findIndex((decision) => decision.id === left.id) - DECISIONS.findIndex((decision) => decision.id === right.id));
    const outputNodes = graph.nodes
      .filter((node) => node.kind === 'output')
      .sort((left, right) => DECISIONS.findIndex((decision) => decision.id === left.id.replace('-output', '')) - DECISIONS.findIndex((decision) => decision.id === right.id.replace('-output', '')));
    const layers = [
      { nodes: featureNodes, x: 10 },
      { nodes: decisionNodes, x: 255 },
      { nodes: outputNodes, x: 500 },
    ];
    const maxLayerSize = Math.max(...layers.map((layer) => layer.nodes.length), 1);
    const rowHeight = 92;
    const layerHeight = maxLayerSize * rowHeight;
    const positions = new Map<string, { x: number; y: number }>();
    for (const layer of layers) {
      const offset = (layerHeight - layer.nodes.length * rowHeight) / 2;
      layer.nodes.forEach((node, index) => positions.set(node.id, { x: layer.x, y: offset + index * rowHeight }));
    }

    return graph.nodes.map((node) => {
      if (node.kind === 'feature') {
        const column = FEATURE_COLUMNS.find((item) => item.key === node.id);
        const change = preview?.changes.find((item) => item.feature === node.id);
        return {
          id: node.id,
          type: 'default',
          className: `provenance-node provenance-feature${change ? ' is-changed' : ''}`,
          data: {
            label: (
              <div className="provenance-node-content">
                <span className="provenance-node-kicker">{change ? 'CORRECTED FEATURE' : 'INPUT FEATURE'}</span>
                <strong>{column?.label ?? auditLabel(node.id)}</strong>
                {change && <span className="provenance-node-value">{displayValue(change.old_value)} <span aria-label="changes to">→</span> {displayValue(change.new_value)}</span>}
              </div>
            ),
          },
          position: positions.get(node.id) ?? { x: 0, y: 0 },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        };
      }

      if (node.kind === 'decision') {
        const decision = DECISIONS.find((item) => item.id === node.id);
        const impact = preview?.decision_impacts.find((item) => item.decision_id === node.id);
        return {
          id: node.id,
          type: 'default',
          className: `provenance-node provenance-decision${impact?.affected ? ' is-affected' : ''}`,
          data: {
            label: (
              <div className="provenance-node-content">
                <span className="provenance-node-kicker">{impact?.affected ? 'AFFECTED BY CORRECTION' : 'DECISION MODEL'}</span>
                <strong>{node.id} · {decision?.label ?? 'Decision'}</strong>
              </div>
            ),
          },
          position: positions.get(node.id) ?? { x: 255, y: 0 },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        };
      }

      const decisionId = node.id.replace(/-output$/, '');
      const decision = DECISIONS.find((item) => item.id === decisionId);
      const impact = preview?.decision_impacts.find((item) => item.decision_id === decisionId);
      const before = preview?.historical_outputs[decisionId];
      const after = preview?.counterfactual_outputs[decisionId];
      return {
        id: node.id,
        type: 'default',
        className: `provenance-node provenance-output${impact?.changed ? ' is-changed' : ''}`,
        data: {
          label: (
            <div className="provenance-node-content">
              <span className="provenance-node-kicker">REPLAYED OUTPUT · {decisionId}</span>
              <strong>{decision?.label ?? 'Decision output'}</strong>
              {before !== undefined && after !== undefined && (
                <span className="provenance-node-value">{before} <span aria-label="changes to">→</span> {after}</span>
              )}
              <span className="provenance-output-status">{impact?.changed ? 'Output changes' : 'Output unchanged'}</span>
            </div>
          ),
        },
        position: positions.get(node.id) ?? { x: 500, y: 0 },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
    });
  }, [graph, preview]);
  const graphEdges = useMemo(() => graph.edges.map((edge) => {
    const source = graph.nodes.find((node) => node.id === edge.source);
    const target = graph.nodes.find((node) => node.id === edge.target);
    const isDecisionDependency = source?.kind === 'decision' && target?.kind === 'decision';
    return {
    id: `${edge.source}-${edge.target}`,
    source: edge.source,
    target: edge.target,
    markerEnd: { type: MarkerType.ArrowClosed },
    label: isDecisionDependency ? 'severity input' : undefined,
    labelStyle: { fill: '#cbd5e1', fontSize: 10, fontWeight: 600 },
    labelBgStyle: { fill: '#0f172a', fillOpacity: 0.92 },
    labelBgPadding: [5, 3] as [number, number],
    style: { stroke: isDecisionDependency ? '#a78bfa' : '#64748b', strokeWidth: isDecisionDependency ? 2.5 : 1.8 },
  };
  }), [graph]);

  const setProposedFeature = (name: keyof EventRecord, value: unknown) => {
    setProposedFeatures((current) => ({ ...current, [name]: value }));
    setPreview(null);
    setGraph({ nodes: [], edges: [] });
    setCorrectionId('');
    setMessage('');
    setAiAnswer('');
    setAiSource('');
    setAiError('');
  };

  const generateDataset = async () => {
    if (!window.confirm('This will create a new 200-record experiment dataset. The current dataset will remain stored for reproducibility. Continue?')) return;
    setBusy(true);
    setError('');
    try {
      const next = await requestJson<DatasetInfo>(`${API_BASE}/api/datasets`, { method: 'POST' });
      setDataset(next);
      setSearch('');
      setSortBy('event_id');
      setDescending(false);
      setPage(1);
      setDetails(null);
      setPreview(null);
      setGraph({ nodes: [], edges: [] });
      setCorrectionId('');
      setRewindResult(null);
      await loadRows(next, 1, '', 'event_id', false);
      await loadAudit(next.dataset_id);
      setMessage(`New 200-record experiment generated without decision labels. Click TRAIN EXPERIMENT to label it with frozen ${next.model_id} ${next.model_version}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create a new dataset.');
    } finally {
      setBusy(false);
    }
  };

  const trainModel = async () => {
    setBusy(true);
    setTraining(true);
    setError('');
    setMessage('');
    try {
      const started = await requestJson<TrainingJob>(`${API_BASE}/api/models/train`, { method: 'POST' });
      setLatestTraining(started);
      let current = started;
      while (current.status === 'RUNNING') {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        current = await requestJson<TrainingJob>(`${API_BASE}/api/models/training/${started.job_id}`);
        setLatestTraining(current);
      }
      await loadModelStatus();
      if (current.status === 'FAILED') throw new Error(current.error || 'Model training failed.');
      setMessage(`Training complete: SOC-MODEL-${current.model_version} trained on ${current.record_count.toLocaleString()} records for ${current.epochs} epochs. Generate a new experiment to use this frozen model.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not train the model.');
    } finally {
      setTraining(false);
      setBusy(false);
    }
  };

  const trainExperiment = async () => {
    if (!dataset || dataset.training_status === 'TRAINED') return;
    setBusy(true);
    setLabeling(true);
    setError('');
    setMessage('');
    try {
      const result = await requestJson<DatasetInfo>(`${API_BASE}/api/datasets/${dataset.dataset_id}/train`, { method: 'POST' });
      setDataset(result);
      await loadRows(result, page, search, sortBy, descending);
      if (details?.current_state.event_id) await loadEvent(result.dataset_id, details.current_state.event_id);
      await loadAudit(result.dataset_id);
      setMessage(`Experiment labeled: D1–D5 predictions generated for ${result.training_record_count} records using frozen ${result.model_id} ${result.model_version}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not label the experiment.');
    } finally {
      setLabeling(false);
      setBusy(false);
    }
  };

  const previewImpact = async () => {
    if (!dataset || !selectedEvent || !changedValues.length) return;
    setBusy(true);
    setError('');
    setMessage('');
    setAiAnswer('');
    setAiSource('');
    setAiError('');
    try {
      const result = await requestJson<PreviewResult>(`${API_BASE}/api/datasets/${dataset.dataset_id}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event_id: selectedEvent.event_id, changes: changedValues }),
      });
      setPreview(result);
      setGraph(result.graph);
      setCorrectionId('');
      setRewindResult(null);
      setDataset((current) => current ? { ...current, workflow_status: 'CORRECTION_PREVIEWED' } : current);
      setMessage(`Preview complete: ${result.affected_decisions.length} decision(s) would change. No event or decision state was modified.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not preview this correction.');
    } finally {
      setBusy(false);
    }
  };

  const applyCorrection = async () => {
    if (!dataset || !selectedEvent || !changedValues.length) return;
    if (!previewMatchesProposal) {
      setError('Preview the current feature changes before applying this correction.');
      return;
    }
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const result = await requestJson<{ correction_id: string }>(`${API_BASE}/api/datasets/${dataset.dataset_id}/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event_id: selectedEvent.event_id, changes: changedValues }),
      });
      setCorrectionId(result.correction_id);
      setDataset((current) => current ? { ...current, workflow_status: 'CORRECTION_APPLIED' } : current);
      setRows((current) => current.map((row) => (
        row.event_id === selectedEvent.event_id ? { ...row, correction_status: 'PROPOSED' } : row
      )));
      await loadAudit(dataset.dataset_id);
      setMessage(`Correction transaction ${result.correction_id} saved with ${changedValues.length} feature change(s). Historical data and decisions remain unchanged until rewind.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save correction.');
    } finally {
      setBusy(false);
    }
  };

  const executeRewind = async () => {
    if (!dataset || !selectedEvent || !correctionId) {
      setError('Apply a correction before running the rewind.');
      return;
    }
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const previewGraph = graph;
      const result = await requestJson<Record<string, unknown>>(`${API_BASE}/api/datasets/${dataset.dataset_id}/rewind`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correction_id: correctionId }),
      });
      await loadEvent(dataset.dataset_id, selectedEvent.event_id);
      setGraph(previewGraph);
      setCorrectionId('');
      await loadRows(dataset, page, search, sortBy, descending);
      await loadAudit(dataset.dataset_id);
      setRewindResult(result);
      setDataset((current) => current ? { ...current, workflow_status: 'VERIFIED' } : current);
      const verified = result.verification as { verification?: string } | undefined;
      const operations = result.rewind_operations as Array<unknown> | undefined;
      setMessage(`Rewind complete: ${operations?.length ?? 0} decision(s) updated; verification ${verified?.verification ?? 'completed'}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not execute the rewind.');
    } finally {
      setBusy(false);
    }
  };

  const cancelCorrection = async () => {
    if (!selectedEvent) return;
    if (dataset && correctionId) {
      setBusy(true);
      try {
        await requestJson(`${API_BASE}/api/datasets/${dataset.dataset_id}/corrections/${correctionId}/cancel`, { method: 'POST' });
        await loadAudit(dataset.dataset_id);
        setDataset((current) => current ? { ...current, workflow_status: 'READY_FOR_CORRECTION' } : current);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Could not cancel the saved correction.');
        setBusy(false);
        return;
      }
      setBusy(false);
    } else if (dataset && preview) {
      setBusy(true);
      try {
        await requestJson(`${API_BASE}/api/datasets/${dataset.dataset_id}/preview/cancel`, { method: 'POST' });
        setDataset((current) => current ? { ...current, workflow_status: 'READY_FOR_CORRECTION' } : current);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Could not cancel the preview.');
        setBusy(false);
        return;
      }
      setBusy(false);
    }
    setProposedFeatures(Object.fromEntries(EDITABLE_FEATURES.map((name) => [name, selectedEvent[name]])));
    setRows((current) => current.map((row) => (
      row.event_id === selectedEvent.event_id ? { ...row, correction_status: 'UNCORRECTED' } : row
    )));
    setPreview(null);
    setGraph({ nodes: [], edges: [] });
    setCorrectionId('');
    setRewindResult(null);
    setMessage('Proposed correction cancelled.');
    setError('');
  };

  const sortColumn = (column: keyof EventRecord) => {
    if (sortBy === column) setDescending((current) => !current);
    else {
      setSortBy(column);
      setDescending(false);
    }
  };

  const askAi = async () => {
    if (!aiQuestion.trim()) return;
    setBusy(true);
    setError('');
    setAiAnswer('');
    setAiSource('');
    setAiError('');
    try {
      const result = await requestJson<{ answer: string; source: string }>(`${API_BASE}/api/ai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: aiQuestion.trim(),
          event_id: selectedEvent?.event_id,
          evidence: {
            experiment_id: dataset?.experiment_id,
            model_id: dataset?.model_id,
            model_version: dataset?.model_version,
            selected_event: selectedEvent ? {
              event_id: selectedEvent.event_id,
              features: Object.fromEntries(
                EDITABLE_FEATURES.map((feature) => [feature, selectedEvent[feature]]),
              ),
              decisions: details?.decisions.map((decision) => ({
                decision_id: decision.decision_id,
                historical_output: decision.historical_output,
                current_output: decision.current_output,
              })),
            } : undefined,
            changes: preview?.changes ?? changedValues,
            affected_decisions: preview?.affected_decisions.map((item) => item.decision_id) ?? [],
          },
        }),
      });
      setAiAnswer(result.answer);
      setAiSource(result.source);
    } catch (cause) {
      setAiError(cause instanceof Error ? cause.message : 'Could not get reasoning.');
    } finally {
      setBusy(false);
    }
  };

  const downloadAudit = () => {
    if (!dataset || !audit.length) return;
    const exportData = {
      dataset_id: dataset.dataset_id,
      exported_at: new Date().toISOString(),
      record_count: audit.length,
      records: audit,
    };
    const file = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${dataset.dataset_id}-audit.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const toggleColumn = (column: string, visible: string[], setVisible: (value: string[]) => void) => {
    setVisible(visible.includes(column) ? visible.filter((item) => item !== column) : [...visible, column]);
  };

  return (
    <div className="app-shell workbench-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">CYBERSECURITY RESEARCH</div>
          <h1>DECISION-REWIND</h1>
        </div>
        <div className="topbar-actions">
          <nav aria-label="Main navigation">
            <a href="#workbench">Dataset Workbench</a>
            <a href="#selected-event">Selected Event</a>
            <a href="#provenance">Provenance</a>
            <a href="#audit">Audit</a>
          </nav>
          <button
            className="secondary retrain-icon-button"
            onClick={trainModel}
            disabled={busy || training || latestTraining?.status === 'RUNNING'}
            aria-label={training || latestTraining?.status === 'RUNNING' ? 'Training global model' : modelStatus?.active_model ? 'Retrain global model' : 'Train global model'}
            title={training || latestTraining?.status === 'RUNNING' ? 'Training global model' : modelStatus?.active_model ? 'Retrain global model' : 'Train global model'}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M20 7v5h-5M4 17v-5h5" />
              <path d="M5.6 9A7 7 0 0 1 18 6.5L20 12M4 12l2 5.5A7 7 0 0 0 18.4 15" />
            </svg>
          </button>
        </div>
      </header>

      <main className="workbench-main">
        <section className="panel dataset-toolbar">
          <div>
            <div className="eyebrow">ACTIVE EXPERIMENT</div>
            <h2>{dataset?.dataset_id ?? (modelStatus ? 'No experiment generated' : 'Loading experiment…')}</h2>
            {dataset && <div className="dataset-meta">
              <span>{dataset?.record_count ?? 0} records</span>
              <span>Seed: {dataset?.random_seed ?? dataset?.seed ?? '—'}</span>
              <span>Created: {dataset ? formatDate(dataset.created_at) : '—'}</span>
              <span>{dataset.experiment_id}</span>
            </div>}
            {dataset && <div className="training-status">
              <strong>Experiment Status:</strong>
              <span className={dataset.training_status === 'TRAINED' ? 'status-trained' : 'status-not-trained'}>
                {dataset.training_status === 'TRAINED' ? 'LABELED' : 'GENERATED / NOT LABELED'}
              </span>
              <span>Workflow: {dataset.workflow_status}</span>
              <span>Frozen model: {dataset.model_id} · {dataset.model_version}</span>
              <span>Global training set: {modelStatus?.models.find((model) => model.model_id === dataset.model_id)?.training_record_count.toLocaleString() ?? '—'} rows · {dataset.training_dataset_id}</span>
            </div>}
          </div>
          <div className="dataset-actions">
            <button className="secondary" onClick={generateDataset} disabled={busy || training || !modelStatus?.active_model}>GENERATE 200-ROW EXPERIMENT</button>
            {dataset && dataset.training_status !== 'TRAINED' && (
              <button onClick={trainExperiment} disabled={busy || training || labeling}>
                {labeling ? 'LABELING…' : 'TRAIN EXPERIMENT'}
              </button>
            )}
          </div>
        </section>

        {message && <div className="workbench-message" role="status">{message}</div>}
        {error && <div className="workbench-error" role="alert">{error}</div>}

        <section className="panel workbench-panel" id="workbench">
          <div className="panel-header">
            <div>
              <div className="eyebrow">200 RECORDS · 20 PER PAGE</div>
              <h2>Dataset Workbench</h2>
            </div>
            <label className="search-control">
              <span>Search records</span>
              <input value={search} onChange={(event) => { setSearch(event.target.value); setPage(1); }} placeholder="Event, user, IP, feature…" />
            </label>
          </div>

          <details className="column-settings">
            <summary>Visible columns</summary>
            <div className="column-toggles">
              <div>
                <strong>Features</strong>
                {FEATURE_COLUMNS.map((column) => (
                  <label key={column.key}>
                    <input type="checkbox" checked={visibleFeatures.includes(column.key)} onChange={() => toggleColumn(column.key, visibleFeatures, setVisibleFeatures)} />
                    {column.label}
                  </label>
                ))}
              </div>
              <div>
                <strong>Decisions</strong>
                {DECISIONS.map((decision) => (
                  <label key={decision.id}>
                    <input type="checkbox" checked={visibleDecisions.includes(decision.id)} onChange={() => toggleColumn(decision.id, visibleDecisions, setVisibleDecisions)} />
                    {decision.id} {decision.label}
                  </label>
                ))}
              </div>
            </div>
          </details>

          <div className="table-scroll">
            <table className="dataset-table">
              <thead>
                <tr>
                  {FEATURE_COLUMNS.filter((column) => visibleFeatures.includes(column.key)).map((column) => (
                    <th key={column.key}>
                      <button className="sort-button" onClick={() => sortColumn(column.key)}>
                        {column.label}{sortBy === column.key ? (descending ? ' ▼' : ' ▲') : ''}
                      </button>
                    </th>
                  ))}
                  {DECISIONS.filter((decision) => visibleDecisions.includes(decision.id)).map((decision) => <th key={decision.id}>{decision.id}</th>)}
                  <th>Correction</th>
                  <th>Impact</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={`${dataset?.dataset_id}-${row.event_id}`} className={selectedEvent?.event_id === row.event_id ? 'selected-row' : ''} onClick={() => dataset && void loadEvent(dataset.dataset_id, row.event_id)}>
                    {FEATURE_COLUMNS.filter((column) => visibleFeatures.includes(column.key)).map((column) => (
                      <td key={column.key}>{displayValue(row[column.key])}</td>
                    ))}
                    {DECISIONS.filter((decision) => visibleDecisions.includes(decision.id)).map((decision) => <td key={decision.id}>{row.decision_outputs?.[decision.id] ?? '—'}</td>)}
                    <td><span className={`table-status ${row.correction_status === 'CORRECTED' ? 'status-corrected' : ''}`}>{row.correction_status}</span></td>
                    <td><span className={`table-status ${row.impact_status === 'AFFECTED' ? 'status-affected' : ''}`}>{row.impact_status}</span></td>
                  </tr>
                ))}
                {!rows.length && <tr><td colSpan={visibleFeatures.length + visibleDecisions.length + 2}>{busy ? 'Loading records…' : 'No matching records.'}</td></tr>}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <span>{totalRows} matching records · Page {page} / {pageCount}</span>
            <div>
              <button className="secondary" disabled={busy || page <= 1} onClick={() => setPage((current) => Math.max(1, current - 1))}>Previous</button>
              <button className="secondary" disabled={busy || page >= pageCount} onClick={() => setPage((current) => Math.min(pageCount, current + 1))}>Next</button>
            </div>
          </div>
        </section>

        <div className="event-analysis-grid">
          <section className="panel selected-event-panel" id="selected-event">
            <div className="panel-header">
              <div>
                <div className="eyebrow">ROW DETAIL</div>
                <h2>Selected Event</h2>
              </div>
              <span className="badge">{selectedEvent?.event_id ?? 'NONE'}</span>
            </div>
            {selectedEvent ? (
              <>
                <div className="feature-detail-grid">
                  {FEATURE_COLUMNS.map((column) => (
                    <div className="feature-detail" key={column.key}>
                      <span>{column.label}</span>
                      <strong>
                        {displayValue(selectedEvent[column.key])}
                        {details?.original_state[column.key] !== selectedEvent[column.key] && (
                          <small className="original-value"> Original: {displayValue(details?.original_state[column.key])}</small>
                        )}
                      </strong>
                    </div>
                  ))}
                </div>
                <h3>Decision State</h3>
                <div className="decision-state-table">
                  {DECISIONS.map((decision) => {
                    const state = selectedDecision(decision.id);
                    const previewDecision = preview?.affected_decisions.concat(preview.unaffected_decisions).find((item) => item.decision_id === decision.id);
                    const hasRecovered = state?.current_output !== state?.historical_output;
                    const decisionStatus = previewDecision
                      ? (previewDecision.affected ? 'AFFECTED' : 'UNCHANGED')
                      : (hasRecovered ? 'RECOVERED' : dataset?.training_status === 'TRAINED' ? 'UNCHANGED' : 'NOT TRAINED');
                    return (
                      <div className="decision-state-row" key={decision.id}>
                        <strong>{decision.id} {decision.label}</strong>
                        <span><small>Historical</small>{state?.historical_output ?? '—'}</span>
                        <span><small>Current</small>{state?.current_output ?? '—'}</span>
                        <span><small>Counterfactual</small>{previewDecision?.counterfactual_output ?? state?.counterfactual_output ?? '—'}</span>
                        <em className={`state ${previewDecision?.affected ? 'affected' : hasRecovered ? 'recovered' : 'safe'}`}>
                          {decisionStatus}
                        </em>
                      </div>
                    );
                  })}
                </div>
                <p className="model-note">
                  {dataset?.training_status === 'TRAINED'
                    ? `Decision model: ${dataset.model_id} · ${dataset.model_version} · trained on dataset ${dataset.dataset_id}`
                    : 'Decision outputs will appear after the dataset is trained.'}
                </p>
              </>
            ) : <p className="placeholder">Select a record in the workbench.</p>}
          </section>

          <section className="panel correction-panel">
            <div className="eyebrow">PRESERVE HISTORY · PREVIEW BEFORE APPLY</div>
            <h2>Correction &amp; Selective Rewind</h2>
            {selectedEvent ? (
              <>
                <div className="editable-feature-list">
                  {EDITABLE_FEATURES.map((name) => {
                    const originalValue = selectedEvent[name];
                    const proposedValue = proposedFeatures[name] ?? originalValue;
                    const isNumber = ['failed_logins', 'login_hour', 'threat_intel_score', 'previous_alerts'].includes(name);
                    return (
                      <div className={`editable-feature ${proposedValue !== originalValue ? 'feature-modified' : ''}`} key={name}>
                        <strong>{name}</strong>
                        <span>Baseline: {displayValue(details?.original_state[name])}</span>
                        {originalValue !== details?.original_state[name] && <span>Current: {displayValue(originalValue)}</span>}
                        {name === 'asset_criticality' ? (
                          <select value={String(proposedValue)} onChange={(event) => setProposedFeature(name, event.target.value)}>
                            {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map((value) => <option key={value}>{value}</option>)}
                          </select>
                        ) : name === 'geo_anomaly' ? (
                          <select value={String(proposedValue)} onChange={(event) => setProposedFeature(name, event.target.value === 'true')}>
                            <option value="true">YES</option><option value="false">NO</option>
                          </select>
                        ) : (
                          <input
                            type={isNumber ? 'number' : 'text'}
                            min={name === 'threat_intel_score' || name === 'login_hour' ? 0 : undefined}
                            max={name === 'threat_intel_score' ? 1 : name === 'login_hour' ? 23 : name === 'failed_logins' ? 20 : name === 'previous_alerts' ? 10 : undefined}
                            step={name === 'threat_intel_score' ? 0.01 : 1}
                            value={String(proposedValue)}
                            onChange={(event) => setProposedFeature(name, isNumber ? Number(event.target.value) : event.target.value)}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
                <p className="model-note">{changedFeatures.length} changed feature(s): {changedFeatures.length ? changedFeatures.join(', ') : 'none'}</p>
                {preview && (
                  <div className="preview-summary">
                    <strong>Impact preview</strong>
                    {preview.changes.map((change) => (
                      <span key={change.feature}>{change.feature}: {displayValue(change.old_value)} → {displayValue(change.new_value)}</span>
                    ))}
                    <span>{preview.affected_decisions.length} affected · {preview.unaffected_decisions.length} unchanged</span>
                    <small>Preview only: the model and persisted event state were not changed.</small>
                    {preview.affected_decisions.map((item) => (
                      <small key={item.decision_id}>
                        {item.decision_id}: {item.current_output} → {item.counterfactual_output}; changed features: {item.changed_features.join(', ')}
                        {item.dependency_paths.map((path) => ` · ${path.join(' → ')}`)}
                      </small>
                    ))}
                  </div>
                )}
                <div className="button-row correction-actions">
                  <button className="secondary" onClick={cancelCorrection} disabled={busy}>CANCEL</button>
                  <button className="secondary" onClick={previewImpact} disabled={busy || dataset?.training_status !== 'TRAINED' || !changedValues.length || Boolean(correctionId)}>{busy ? 'WORKING…' : 'PREVIEW IMPACT'}</button>
                  <button onClick={applyCorrection} disabled={busy || dataset?.training_status !== 'TRAINED' || !changedValues.length || !previewMatchesProposal || Boolean(correctionId)}>APPLY CORRECTION</button>
                  <button className="rewind-button" onClick={executeRewind} disabled={busy || dataset?.training_status !== 'TRAINED' || !correctionId}>
                    {busy ? 'WORKING…' : 'REWIND DECISION'}
                  </button>
                </div>
                {correctionId && <p className="action-message">Saved correction {correctionId}. Run rewind to update affected decisions.</p>}
                {rewindResult && <p className="verification-message">Verification: {(rewindResult.verification as { verification?: string })?.verification ?? '—'}</p>}
              </>
            ) : <p className="placeholder">Select a record to propose a correction.</p>}
          </section>
        </div>

        <section className="panel graph-panel" id="provenance">
          <div className="panel-header">
            <div>
              <div className="eyebrow">FOLLOW THE CORRECTION</div>
              <h2>Provenance Graph</h2>
              <p className="provenance-intro">
                Read left to right: event features feed decision models, which produce outcomes.
                Corrected inputs and changed outcomes are highlighted.
              </p>
            </div>
          </div>
          {preview && (
            <div className="provenance-summary" role="status">
              <strong>{preview.changed_features.length} corrected feature{preview.changed_features.length === 1 ? '' : 's'}</strong>
              <span>→</span>
              <strong>{preview.affected_decisions.length} affected decision{preview.affected_decisions.length === 1 ? '' : 's'}</strong>
              <span>·</span>
              <span>Event {preview.event_id}</span>
            </div>
          )}
          <div className="provenance-legend" aria-label="Graph legend">
            <span><i className="legend-swatch feature-swatch" /> Event feature</span>
            <span><i className="legend-swatch decision-swatch" /> Decision model</span>
            <span><i className="legend-swatch output-swatch" /> Decision output</span>
            <span><i className="legend-swatch changed-swatch" /> Changed by correction</span>
            <span><i className="legend-line" /> Dependency / data flow</span>
          </div>
          <div className={`graph-box workbench-graph${graphNodes.length ? '' : ' is-empty'}`}>
            {graphNodes.length ? (
              <ReactFlow
                key={`${dataset?.dataset_id ?? 'empty'}-${graphNodes.map((node) => node.id).join('-')}`}
                nodes={graphNodes}
                edges={graphEdges}
                fitView
                fitViewOptions={{ padding: 0.12, minZoom: 0.25, maxZoom: 1 }}
                onInit={(instance) => window.requestAnimationFrame(() => {
                  window.requestAnimationFrame(() => instance.fitView({ padding: 0.12, minZoom: 0.25, maxZoom: 1 }));
                })}
                nodesDraggable={false}
                elementsSelectable={false}
                panOnDrag
                zoomOnScroll
              >
                <Background color="#1f4e79" gap={18} />
                <Controls />
              </ReactFlow>
            ) : <p className="placeholder">Preview impact to view provenance paths for this event’s proposed changes.</p>}
          </div>
        </section>

        <div className="bottom-panels">
          <section className="panel ai-panel">
            <h2>AI Reasoning</h2>
            <div className="ai-question">
              <input
                value={aiQuestion}
                onChange={(event) => {
                  setAiQuestion(event.target.value);
                  setAiError('');
                  setAiAnswer('');
                  setAiSource('');
                }}
                onKeyDown={(event) => { if (event.key === 'Enter') void askAi(); }}
              />
              <button onClick={askAi} disabled={busy || !aiQuestion.trim()}>ASK</button>
            </div>
            {aiError && <p className="workbench-error" role="alert">{aiError}</p>}
            <p className="ai-answer">{aiAnswer || 'Ask any question—experiment context is included when available.'}</p>
            {aiSource && <small className="ai-source">Answered by {aiSource}</small>}
          </section>
          <section className="panel audit-panel" id="audit">
            <div className="audit-heading">
              <div>
                <div className="eyebrow">PERSISTED DATASET HISTORY</div>
                <h2>Audit Trail</h2>
              </div>
              <button className="secondary audit-download" onClick={downloadAudit} disabled={!dataset || !audit.length}>
                DOWNLOAD JSON
              </button>
            </div>
            <div className="timeline">
              {audit.length ? audit.slice(0, 12).map((entry) => (
                <div className="timeline-item" key={entry.audit_id}>
                  <div className="audit-record-header">
                    <strong>{auditLabel(entry.operation)}</strong>
                    <time dateTime={entry.timestamp}>{formatDate(entry.timestamp)}</time>
                  </div>
                  <span className="audit-event-id">{entry.event_id || dataset?.dataset_id}</span>
                  <div className="audit-record-details"><AuditValue value={entry.details} /></div>
                </div>
              )) : <p>No audit records for this dataset.</p>}
              {audit.length > 12 && <p className="audit-count">Showing 12 most recent of {audit.length} records. Download JSON for the full audit trail.</p>}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
