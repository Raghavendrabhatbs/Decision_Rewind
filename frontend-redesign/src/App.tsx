import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, { Background, Controls, MarkerType, Position } from 'reactflow';
import 'reactflow/dist/style.css';
import { marked } from 'marked';

marked.setOptions({
  gfm: true,
  breaks: true,
});

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

type AIReasoningLogEvent = {
  id: string;
  timestamp: string;
  event_type: string;
  source: string;
  details: Record<string, unknown>;
};

type AIReasoningEvidence = {
  experiment: Record<string, unknown> | null;
  event: { event_id: string } | null;
  correction: { correction_id: string; status: string } | null;
  decisions: {
    historical: Record<string, string>;
    current: Record<string, string>;
    counterfactual: Record<string, string | null>;
  } | null;
  affected_decisions: string[];
  provenance: { nodes: Array<{ id: string; kind: string }>; edges: Array<{ source: string; target: string }> } | null;
  recovery: Record<string, unknown> | null;
  verification: Record<string, unknown> | null;
  universal_log_context: {
    event_count: number;
    returned_event_count: number;
    truncated: boolean;
    events: AIReasoningLogEvent[];
  };
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
  const [showWorkbench, setShowWorkbench] = useState(false);
  const [showOverview, setShowOverview] = useState(false);
  const [showExperimentReady, setShowExperimentReady] = useState(false);
  const [showGenerateDialog, setShowGenerateDialog] = useState(false);
  const [workbenchView, setWorkbenchView] = useState<'records' | 'event' | 'rewind' | 'ai' | 'audit'>('records');
  const [isLeavingHero, setIsLeavingHero] = useState(false);
  const [trainingMessageIndex, setTrainingMessageIndex] = useState(0);
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
  const [aiEvidence, setAiEvidence] = useState<AIReasoningEvidence | null>(null);
  const [aiError, setAiError] = useState('');
  const [aiQuestion, setAiQuestion] = useState('Why would this correction affect these decisions?');
  const formattedAiAnswer = useMemo(() => {
    if (!aiAnswer) return '';
    try {
      return marked.parse(aiAnswer) as string;
    } catch {
      return aiAnswer;
    }
  }, [aiAnswer]);
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
    setAiEvidence(null);
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
    if (!training) {
      setTrainingMessageIndex(0);
      return;
    }
    const interval = window.setInterval(() => {
      setTrainingMessageIndex((current) => (current + 1) % 4);
    }, 2200);
    return () => window.clearInterval(interval);
  }, [training]);

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
      { nodes: featureNodes, x: 20 },
      { nodes: decisionNodes, x: 310 },
      { nodes: outputNodes, x: 600 },
    ];
    const maxLayerSize = Math.max(...layers.map((layer) => layer.nodes.length), 1);
    const rowHeight = 98;
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
                <div className="provenance-node-header">
                  <span className="provenance-node-kicker">{change ? '⚡ CORRECTED' : 'INPUT SIGNAL'}</span>
                  {change ? (
                    <span className="provenance-chip changed">MODIFIED</span>
                  ) : null}
                </div>
                <strong>{column?.label ?? auditLabel(node.id)}</strong>
                {change ? (
                  <div className="provenance-node-diff">
                    <span className="diff-old">{displayValue(change.old_value)}</span>
                    <span className="diff-arrow">→</span>
                    <span className="diff-new">{displayValue(change.new_value)}</span>
                  </div>
                ) : (
                  <span className="provenance-node-sub">{node.id}</span>
                )}
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
                <div className="provenance-node-header">
                  <span className="provenance-node-kicker">{impact?.affected ? '🔥 AFFECTED' : 'DECISION ENGINE'}</span>
                  <span className={`provenance-chip ${impact?.affected ? 'affected' : 'safe'}`}>
                    {impact?.affected ? 'DIVERGED' : 'STABLE'}
                  </span>
                </div>
                <strong>{node.id} · {decision?.label ?? 'Decision'}</strong>
                <span className="provenance-node-sub">
                  {impact?.affected ? 'Output recomputed' : 'Inputs unchanged'}
                </span>
              </div>
            ),
          },
          position: positions.get(node.id) ?? { x: 310, y: 0 },
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
              <div className="provenance-node-header">
                <span className="provenance-node-kicker">OUTCOME · {decisionId}</span>
                <span className={`provenance-chip ${impact?.changed ? 'changed' : 'safe'}`}>
                  {impact?.changed ? 'CHANGED' : 'MATCH'}
                </span>
              </div>
              <strong>{decision?.label ?? 'Decision output'}</strong>
              {before !== undefined && after !== undefined ? (
                <div className="provenance-node-diff">
                  <span className={`diff-val ${impact?.changed ? 'diff-old' : ''}`}>{before}</span>
                  {impact?.changed && (
                    <>
                      <span className="diff-arrow">→</span>
                      <span className="diff-new">{after}</span>
                    </>
                  )}
                </div>
              ) : null}
            </div>
          ),
        },
        position: positions.get(node.id) ?? { x: 600, y: 0 },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
    });
  }, [graph, preview]);

  const graphEdges = useMemo(() => graph.edges.map((edge) => {
    const source = graph.nodes.find((node) => node.id === edge.source);
    const target = graph.nodes.find((node) => node.id === edge.target);
    const isDecisionDependency = source?.kind === 'decision' && target?.kind === 'decision';

    const isSourceChanged = preview?.changes.some((c) => c.feature === edge.source);
    const isTargetAffected = preview?.decision_impacts.some((d) => d.decision_id === edge.target && d.affected);
    const isOutputChanged = edge.target.endsWith('-output') && preview?.decision_impacts.some((d) => d.decision_id === edge.target.replace('-output', '') && d.changed);
    const isAffectedCascade = Boolean(isSourceChanged || isTargetAffected || isOutputChanged);

    let strokeColor = '#94a3b8';
    let strokeWidth = 1.6;
    let animated = false;
    let label = undefined;

    if (isDecisionDependency) {
      label = 'severity input';
      strokeColor = isTargetAffected ? '#dc2626' : '#4f46e5';
      strokeWidth = 2.4;
      animated = true;
    } else if (isAffectedCascade) {
      strokeColor = '#dc2626';
      strokeWidth = 2.5;
      animated = true;
    }

    return {
      id: `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      animated,
      markerEnd: { type: MarkerType.ArrowClosed, color: strokeColor },
      label,
      labelStyle: { fill: strokeColor, fontSize: 10, fontWeight: 700, letterSpacing: '0.04em' },
      labelBgStyle: { fill: '#ffffff', fillOpacity: 0.96 },
      labelBgPadding: [6, 4] as [number, number],
      style: {
        stroke: strokeColor,
        strokeWidth,
      },
    };
  }), [graph, preview]);

  const setProposedFeature = (name: keyof EventRecord, value: unknown) => {
    setProposedFeatures((current) => ({ ...current, [name]: value }));
    setPreview(null);
    setGraph({ nodes: [], edges: [] });
    setCorrectionId('');
    setMessage('');
    setAiAnswer('');
    setAiSource('');
    setAiEvidence(null);
    setAiError('');
  };

  const generateDataset = async () => {
    setShowGenerateDialog(false);
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
      setShowExperimentReady(false);
      setShowOverview(false);
      setShowWorkbench(true);
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
      setShowExperimentReady(true);
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
    setAiEvidence(null);
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
    setAiEvidence(null);
    setAiError('');
    try {
      const result = await requestJson<{ answer: string; source: string; evidence: AIReasoningEvidence }>(`${API_BASE}/api/ai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: aiQuestion.trim(),
          dataset_id: dataset?.dataset_id,
          experiment_id: dataset?.experiment_id,
          event_id: selectedEvent?.event_id,
          correction_id: correctionId || undefined,
        }),
      });
      setAiAnswer(result.answer);
      setAiSource(result.source);
      setAiEvidence(result.evidence);
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

  const enterWorkbench = () => {
    setIsLeavingHero(true);
    window.setTimeout(() => setShowOverview(true), 650);
  };

  const isModelTraining = training || latestTraining?.status === 'RUNNING';
  const modelReady = Boolean(modelStatus?.active_model);
  const activePageMode = !showWorkbench
    ? (showOverview ? (showExperimentReady ? 'ready' : 'overview') : 'hero')
    : 'workbench';

  return (
    <div className={`app-root mode-${activePageMode}`}>
      {/* Global persistent background video across all pages */}
      <div className="global-bg-container" aria-hidden="true">
        <video
          className="global-video"
          autoPlay
          muted
          loop
          playsInline
        >
          <source src="/hero-background.mp4" type="video/mp4" />
          <source src="/videoplayback.mp4" type="video/mp4" />
        </video>
        <div className={`global-wash mode-${activePageMode}`} />
        <div className="global-grid" />
      </div>

      {!showWorkbench ? (
        showOverview ? (
          showExperimentReady ? (
            <main className="experiment-ready-page">
              <header className="overview-header experiment-ready-header">
                <button className="overview-brand" onClick={() => { setIsLeavingHero(false); setShowOverview(false); setShowExperimentReady(false); }} aria-label="Return to hero">
                  <img src="/decision-rewind-logo.png" alt="" />
                </button>
              </header>
              <section className="experiment-ready-content">
                <div className="success-mark" aria-hidden="true">
                  <svg viewBox="0 0 64 64">
                    <path d="m17 33 10 10 20-22" />
                  </svg>
                </div>
                <p className="overview-kicker">GLOBAL MODEL // READY</p>
                <h1>Model trained<br /><span>successfully.</span></h1>
                <p className="overview-copy">
                  Your global model has learned from 20,000 events and is ready to power a fresh decision experiment.
                </p>
                <div className="ready-card">
                  <div><span>MODEL</span><strong>SOC-MODEL-{latestTraining?.model_version ?? modelStatus?.active_model?.model_version ?? '—'}</strong></div>
                  <div><span>TRAINING SET</span><strong>{(latestTraining?.record_count ?? modelStatus?.training_record_count ?? 20000).toLocaleString()} EVENTS</strong></div>
                  <div><span>STATUS</span><strong>FROZEN &amp; READY</strong></div>
                </div>
                <button className="overview-train-button generate-button" onClick={() => setShowGenerateDialog(true)} disabled={busy}>
                  GENERATE 200-ROW EXPERIMENT <span className="overview-button-arrow">↗</span>
                </button>
                <p className="ready-note">A new unlabeled experiment will be created without changing the stored training data.</p>
              </section>
              {showGenerateDialog && (
                <div className="generate-modal-backdrop" role="presentation">
                  <section className="generate-modal" role="dialog" aria-modal="true" aria-labelledby="generate-modal-title-ready">
                    <div className="generate-modal-mark" aria-hidden="true">↗</div>
                    <p className="eyebrow">NEW EXPERIMENT // CONFIRM</p>
                    <h2 id="generate-modal-title-ready">Create a fresh<br /><span>200-record experiment?</span></h2>
                    <p>
                      A new unlabeled experiment will be generated from the frozen global model.
                      Your current dataset will remain stored for reproducibility.
                    </p>
                    <div className="generate-modal-actions">
                      <button className="secondary" onClick={() => setShowGenerateDialog(false)}>CANCEL</button>
                      <button onClick={() => void generateDataset()}>GENERATE EXPERIMENT <span>↗</span></button>
                    </div>
                  </section>
                </div>
              )}
            </main>
          ) : (
            <main className="overview-page">
              <header className="overview-header">
                <button className="overview-brand" onClick={() => { setIsLeavingHero(false); setShowOverview(false); }} aria-label="Return to hero">
                  <img src="/decision-rewind-logo.png" alt="" />
                </button>
              </header>

              <section className="overview-content">
                <p className="overview-kicker">DECISION INTELLIGENCE PLATFORM</p>
                <h1>Understand the past.<br /><span>Change what comes next.</span></h1>
                <p className="overview-copy">
                  Decision Rewind lets you revisit consequential decisions, trace the evidence behind them,
                  and simulate how a corrected signal changes the outcome.
                </p>
                <div className="overview-stats">
                  <div><strong>20,000</strong><span>GLOBAL EVENTS</span></div>
                  <div><strong>05</strong><span>DECISION LAYERS</span></div>
                  <div><strong>01</strong><span>REWIND ENGINE</span></div>
                </div>
                <button
                  className="overview-train-button"
                  onClick={() => void trainModel()}
                  disabled={busy || isModelTraining}
                >
                  {isModelTraining ? 'TRAINING MODEL' : modelReady ? 'RETRAIN GLOBAL MODEL' : 'TRAIN GLOBAL MODEL'}
                  <span className="overview-button-arrow">↗</span>
                </button>
                <button className="overview-enter" onClick={() => setShowWorkbench(true)} disabled={!modelReady}>
                  ENTER WORKBENCH <span>→</span>
                </button>
              </section>

              {isModelTraining && (
                <div className={`training-overlay training-state-${trainingMessageIndex}`} role="status" aria-live="polite">
                  <div className="training-orbit">
                    <span /><span /><span />
                  </div>
                  <p className="training-kicker">GLOBAL MODEL // PROCESSING</p>
                  <h2>
                    {[
                      <>Mapping the signals<br />behind every decision.</>,
                      <>Finding the patterns<br />inside the noise.</>,
                      <>Connecting evidence<br />across the system.</>,
                      <>Learning the shape<br />of every decision.</>,
                    ][trainingMessageIndex]}
                  </h2>
                  <div className="training-progress">
                    <span />
                  </div>
                  <p className="training-detail">
                    {latestTraining?.current_epoch ? `EPOCH ${latestTraining.current_epoch} / ${latestTraining.epochs}` : 'INITIALIZING TRAINING DATA'}
                  </p>
                </div>
              )}
            </main>
          )
        ) : (
          <main className={`hero-landing${isLeavingHero ? ' is-leaving' : ''}`}>
            <section className="hero-content">
              <div className="hero-mark">
                <img
                  src="/decision-rewind-logo.png"
                  alt="Decision Rewind"
                  className="hero-logo"
                />
              </div>
              <button className="hero-cta" onClick={enterWorkbench} disabled={isLeavingHero}>
                <span>REWIND NOW</span>
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M5 12h13M13 6l6 6-6 6" />
                </svg>
              </button>
            </section>
          </main>
        )
      ) : (
        <div className="app-shell workbench-shell">
          <header className="topbar">
            <button
              className="workbench-brand"
              onClick={() => { setShowWorkbench(false); setShowOverview(false); setShowExperimentReady(false); setIsLeavingHero(false); }}
              aria-label="Return to hero"
            >
              <img src="/decision-rewind-logo.png" alt="Decision Rewind" />
              <span>
                <small>DECISION INTELLIGENCE</small>
                <strong>WORKBENCH</strong>
              </span>
            </button>
            <div className="topbar-actions">
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

          <nav className="workbench-navigation-bar" aria-label="Workbench Stages">
            <button
              type="button"
              className={`workbench-tab ${workbenchView === 'records' ? 'is-active' : ''}`}
              onClick={() => setWorkbenchView('records')}
            >
              <span className="tab-idx">01</span>
              <span className="tab-label">Dataset Explorer</span>
              <span className="tab-count">{totalRows || (dataset?.record_count ?? 200)}</span>
            </button>
            <button
              type="button"
              className={`workbench-tab ${workbenchView === 'event' ? 'is-active' : ''}`}
              onClick={() => setWorkbenchView('event')}
            >
              <span className="tab-idx">02</span>
              <span className="tab-label">Event &amp; Decisions</span>
              <span className="tab-count">{selectedEvent?.event_id ?? 'None'}</span>
            </button>
            <button
              type="button"
              className={`workbench-tab ${workbenchView === 'rewind' ? 'is-active' : ''}`}
              onClick={() => setWorkbenchView('rewind')}
            >
              <span className="tab-idx">03</span>
              <span className="tab-label">Correction &amp; Provenance</span>
              {changedFeatures.length > 0 && <span className="tab-count badge-alert">{changedFeatures.length} edit</span>}
              {preview && <span className="tab-count badge-live">{preview.affected_decisions.length} impact</span>}
            </button>
            <button
              type="button"
              className={`workbench-tab ${workbenchView === 'ai' ? 'is-active' : ''}`}
              onClick={() => setWorkbenchView('ai')}
            >
              <span className="tab-idx">04</span>
              <span className="tab-label">AI Reasoning</span>
              <span className="tab-count">Copilot</span>
            </button>
            <button
              type="button"
              className={`workbench-tab ${workbenchView === 'audit' ? 'is-active' : ''}`}
              onClick={() => setWorkbenchView('audit')}
            >
              <span className="tab-idx">05</span>
              <span className="tab-label">Audit Trail</span>
              <span className="tab-count">{audit.length}</span>
            </button>
          </nav>

      <main className="workbench-main">
        {message && <div className="workbench-message" role="status">{message}</div>}
        {error && <div className="workbench-error" role="alert">{error}</div>}

        {/* =================================================================
            VIEW 01: DATASET EXPLORER
            ================================================================= */}
        {workbenchView === 'records' && (
          <div className="view-stage view-stage-records">
            <section className="panel dataset-toolbar">
              <div>
                <div className="eyebrow">ACTIVE EXPERIMENT // OVERVIEW</div>
                <h2>{dataset?.dataset_id ?? (modelStatus ? 'No experiment generated' : 'Loading experiment…')}</h2>
                {dataset && (
                  <div className="dataset-meta">
                    <span>{dataset?.record_count ?? 0} records</span>
                    <span>Seed: {dataset?.random_seed ?? dataset?.seed ?? '—'}</span>
                    <span>Created: {formatDate(dataset.created_at)}</span>
                    <span>{dataset.experiment_id}</span>
                  </div>
                )}
                {dataset && (
                  <div className="training-status">
                    <strong>Experiment Status:</strong>
                    <span className={dataset.training_status === 'TRAINED' ? 'status-trained' : 'status-not-trained'}>
                      {dataset.training_status === 'TRAINED' ? 'LABELED' : 'GENERATED / NOT LABELED'}
                    </span>
                    <span>Workflow: {dataset.workflow_status}</span>
                    <span>Frozen model: {dataset.model_id} · {dataset.model_version}</span>
                    <span>Global training set: {modelStatus?.models.find((model) => model.model_id === dataset.model_id)?.training_record_count.toLocaleString() ?? '—'} rows</span>
                  </div>
                )}
              </div>
              <div className="dataset-actions">
                <button className="secondary" onClick={() => setShowGenerateDialog(true)} disabled={busy || training || !modelStatus?.active_model}>
                  GENERATE 200-ROW EXPERIMENT ↗
                </button>
                {dataset && dataset.training_status !== 'TRAINED' && (
                  <button onClick={trainExperiment} disabled={busy || training || labeling}>
                    {labeling ? 'LABELING EXPERIMENT…' : 'TRAIN EXPERIMENT ↗'}
                  </button>
                )}
              </div>
            </section>

            <section className="panel workbench-panel" id="workbench">
              <div className="panel-header">
                <div>
                  <div className="eyebrow">STAGE 01 // EVENT EXPLORER</div>
                  <h2>Dataset Records</h2>
                  <p className="stage-description">
                    Browse all telemetry events generated from your frozen model. Select any event to inspect its 5-layer decisions or propose a signal correction.
                  </p>
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
                      <tr
                        key={`${dataset?.dataset_id}-${row.event_id}`}
                        className={selectedEvent?.event_id === row.event_id ? 'selected-row' : ''}
                        onClick={() => dataset && void loadEvent(dataset.dataset_id, row.event_id)}
                      >
                        {FEATURE_COLUMNS.filter((column) => visibleFeatures.includes(column.key)).map((column) => (
                          <td key={column.key}>{displayValue(row[column.key])}</td>
                        ))}
                        {DECISIONS.filter((decision) => visibleDecisions.includes(decision.id)).map((decision) => (
                          <td key={decision.id}>{row.decision_outputs?.[decision.id] ?? '—'}</td>
                        ))}
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

            {selectedEvent && (
              <div className="selected-event-dock">
                <div className="selected-dock-info">
                  <span className="badge">SELECTED EVENT</span>
                  <strong>{selectedEvent.event_id}</strong>
                  <span>User: {selectedEvent.user_id}</span>
                  <span>IP: {selectedEvent.source_ip}</span>
                  <span>Criticality: {selectedEvent.asset_criticality}</span>
                  <span>Threat Score: {selectedEvent.threat_intel_score}</span>
                </div>
                <button onClick={() => setWorkbenchView('event')}>
                  ANALYZE EVENT &amp; DECISIONS <span>→</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* =================================================================
            VIEW 02: EVENT ANALYSIS & DECISIONS
            ================================================================= */}
        {workbenchView === 'event' && (
          <div className="view-stage view-stage-event">
            <section className="panel selected-event-panel">
              <div className="panel-header">
                <div>
                  <div className="eyebrow">STAGE 02 // EVENT TELEMETRY &amp; DECISION LAYERS</div>
                  <h2>Selected Event: {selectedEvent?.event_id ?? 'None'}</h2>
                  <p className="stage-description">
                    Inspect all 12 input features and evaluate the 5 sequential decision layers (Authentication → Threat Severity → Asset Protection → Incident Escalation → Response Action).
                  </p>
                </div>
                <span className="badge">{selectedEvent?.event_id ?? 'NO EVENT SELECTED'}</span>
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

                  <h3>5-Layer Decision Pipeline</h3>
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
              ) : (
                <div className="empty-state-card">
                  <p>No event currently selected. Return to the Dataset Explorer to pick a record.</p>
                  <button className="secondary" onClick={() => setWorkbenchView('records')}>GO TO DATASET EXPLORER →</button>
                </div>
              )}

              <div className="view-navigation-footer">
                <button className="secondary" onClick={() => setWorkbenchView('records')}>
                  ← BACK TO DATASET EXPLORER
                </button>
                {selectedEvent && (
                  <button onClick={() => setWorkbenchView('rewind')}>
                    SIMULATE CORRECTION &amp; PROVENANCE <span>→</span>
                  </button>
                )}
              </div>
            </section>
          </div>
        )}

        {/* =================================================================
            VIEW 03: CORRECTION & PROVENANCE GRAPH (COMBINED VIEW)
            ================================================================= */}
        {workbenchView === 'rewind' && (
          <div className="view-stage view-stage-rewind-combined">
            <div className="rewind-provenance-layout">
              {/* LEFT COLUMN: COUNTERFACTUAL CORRECTION & REWIND */}
              <section className="panel correction-panel">
                <div className="panel-header">
                  <div>
                    <div className="eyebrow">STAGE 03 // COUNTERFACTUAL REWIND</div>
                    <h2>Signal Correction &amp; Rewind</h2>
                    <p className="stage-description">
                      Modify telemetry signals to test counterfactual scenarios. Preview affected decisions without mutating historical data, then selectively rewind to update decisions.
                    </p>
                  </div>
                  {selectedEvent && <span className="badge">EVENT {selectedEvent.event_id}</span>}
                </div>

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
                        <strong>Impact Preview Analysis</strong>
                        {preview.changes.map((change) => (
                          <span key={change.feature}>{change.feature}: {displayValue(change.old_value)} → {displayValue(change.new_value)}</span>
                        ))}
                        <span>{preview.affected_decisions.length} affected decision(s) · {preview.unaffected_decisions.length} unchanged</span>
                        <small>Counterfactual Preview: Persisted event state remains unchanged until rewind is executed.</small>
                        {preview.affected_decisions.map((item) => (
                          <small key={item.decision_id}>
                            {item.decision_id}: {item.current_output} → {item.counterfactual_output}; changed features: {item.changed_features.join(', ')}
                            {item.dependency_paths.map((path) => ` · ${path.join(' → ')}`)}
                          </small>
                        ))}
                      </div>
                    )}

                    <div className="button-row correction-actions">
                      <button className="secondary" onClick={cancelCorrection} disabled={busy}>CANCEL CHANGES</button>
                      <button className="secondary" onClick={previewImpact} disabled={busy || dataset?.training_status !== 'TRAINED' || !changedValues.length || Boolean(correctionId)}>
                        {busy ? 'WORKING…' : 'PREVIEW IMPACT ↗'}
                      </button>
                      <button onClick={applyCorrection} disabled={busy || dataset?.training_status !== 'TRAINED' || !changedValues.length || !previewMatchesProposal || Boolean(correctionId)}>
                        APPLY CORRECTION
                      </button>
                      <button className={`rewind-button ${correctionId ? 'is-ready' : ''}`} onClick={executeRewind} disabled={busy || dataset?.training_status !== 'TRAINED' || !correctionId}>
                        {busy ? 'WORKING…' : 'REWIND DECISION ↺'}
                      </button>
                    </div>
                    {correctionId && <p className="action-message">Saved correction transaction {correctionId}. Run rewind to selectively recompute affected decisions.</p>}
                    {rewindResult && <p className="verification-message">Verification: {(rewindResult.verification as { verification?: string })?.verification ?? '—'}</p>}
                  </>
                ) : (
                  <div className="empty-state-card">
                    <p>Select a record first in the Dataset Explorer to simulate corrections.</p>
                    <button className="secondary" onClick={() => setWorkbenchView('records')}>GO TO DATASET EXPLORER →</button>
                  </div>
                )}
              </section>

              {/* RIGHT COLUMN: CAUSAL PROVENANCE GRAPH */}
              <section className="panel graph-panel">
                <div className="panel-header">
                  <div>
                    <div className="eyebrow">STAGE 03 // CAUSAL PROVENANCE GRAPH</div>
                    <h2>Data Lineage &amp; Decision Cascades</h2>
                    <p className="provenance-intro">
                      Read left to right: telemetry features feed decision models, producing outcomes. Corrected inputs and changed decisions are highlighted in real time.
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

                <div className="provenance-flow-banner" aria-hidden="true">
                  <div className="flow-step-col">
                    <span className="flow-step-badge">STEP 1</span>
                    <div className="flow-step-info">
                      <strong>TELEMETRY SIGNALS</strong>
                      <small>Raw Event Inputs</small>
                    </div>
                  </div>
                  <div className="flow-step-arrow">→</div>
                  <div className="flow-step-col">
                    <span className="flow-step-badge">STEP 2</span>
                    <div className="flow-step-info">
                      <strong>DECISION ENGINES</strong>
                      <small>Sequential Logic</small>
                    </div>
                  </div>
                  <div className="flow-step-arrow">→</div>
                  <div className="flow-step-col">
                    <span className="flow-step-badge">STEP 3</span>
                    <div className="flow-step-info">
                      <strong>REPLAYED OUTCOMES</strong>
                      <small>Counterfactual Impact</small>
                    </div>
                  </div>
                </div>

                <div className={`graph-box workbench-graph${graphNodes.length ? '' : ' is-empty'}`}>
                  {graphNodes.length ? (
                    <ReactFlow
                      key={`${dataset?.dataset_id ?? 'empty'}-${graphNodes.map((node) => node.id).join('-')}`}
                      nodes={graphNodes}
                      edges={graphEdges}
                      fitView
                      fitViewOptions={{ padding: 0.15, minZoom: 0.25, maxZoom: 1.1 }}
                      onInit={(instance) => window.requestAnimationFrame(() => {
                        window.requestAnimationFrame(() => instance.fitView({ padding: 0.15, minZoom: 0.25, maxZoom: 1.1 }));
                      })}
                      nodesDraggable={false}
                      elementsSelectable={false}
                      panOnDrag
                      zoomOnScroll
                    >
                      <Background color="#d4d4d8" gap={20} size={1.5} />
                      <Controls />
                    </ReactFlow>
                  ) : (
                    <div className="provenance-empty-guide">
                      <div className="empty-guide-icon" aria-hidden="true">↺</div>
                      <h4>No Active Simulation</h4>
                      <p>
                        {selectedEvent
                          ? 'Modify one or more features on the left and click "PREVIEW IMPACT ↗" to trace the causal graph and decision cascades.'
                          : 'Select an event from the Dataset Explorer, adjust features, and click Preview Impact.'}
                      </p>
                    </div>
                  )}
                </div>
              </section>
            </div>

            <div className="view-navigation-footer">
              <button className="secondary" onClick={() => setWorkbenchView('event')}>
                ← BACK TO EVENT ANALYSIS
              </button>
              <button onClick={() => setWorkbenchView('ai')}>
                PROCEED TO AI REASONING <span>→</span>
              </button>
            </div>
          </div>
        )}

        {/* =================================================================
            VIEW 04: AI REASONING
            ================================================================= */}
        {workbenchView === 'ai' && (
          <div className="view-stage view-stage-ai">
            <section className="panel ai-panel">
              <div className="panel-header">
                <div>
                  <div className="eyebrow">STAGE 04 // AI DECISION COPILOT</div>
                  <h2>AI Evidentiary Reasoning</h2>
                  <p className="stage-description">
                    Ask natural language questions about decisions, causal relationships, or system logs. Answers are grounded in deterministic state, counterfactual diffs, and universal event history.
                  </p>
                </div>
              </div>

              <div className="ai-suggestions-row">
                <span className="eyebrow">Suggested queries:</span>
                <button type="button" className="suggestion-chip" onClick={() => setAiQuestion('Why would this correction affect these decisions?')}>
                  Why would this correction affect decisions? ↗
                </button>
                <button type="button" className="suggestion-chip" onClick={() => setAiQuestion('Explain the causal path from threat score to Response Action.')}>
                  Explain cascade to Response Action ↗
                </button>
                <button type="button" className="suggestion-chip" onClick={() => setAiQuestion('What was the deterministic verification result for the latest rewind?')}>
                  Check verification outcome ↗
                </button>
                <button type="button" className="suggestion-chip" onClick={() => setAiQuestion('Summarize recent audit log events for this dataset.')}>
                  Summarize audit logs ↗
                </button>
              </div>

              <div className="ai-question">
                <input
                  aria-label="Ask AI reasoning"
                  value={aiQuestion}
                  placeholder="Ask about decisions, events, causal graphs, or application logs…"
                  onChange={(event) => {
                    setAiQuestion(event.target.value);
                    setAiError('');
                    setAiAnswer('');
                    setAiSource('');
                    setAiEvidence(null);
                  }}
                  onKeyDown={(event) => { if (event.key === 'Enter') void askAi(); }}
                />
                <button onClick={askAi} disabled={busy || !aiQuestion.trim()}>{busy ? 'THINKING…' : 'ASK AI ↗'}</button>
              </div>

              {aiError && <p className="workbench-error" role="alert">{aiError}</p>}
              {aiAnswer ? (
                <div
                  className="ai-answer-rendered"
                  dangerouslySetInnerHTML={{ __html: formattedAiAnswer }}
                />
              ) : (
                <p className="ai-answer">Ask about decisions, experiments, or application history and logs.</p>
              )}
              {aiSource && <small className="ai-source">Answered by {aiSource}</small>}

              {aiEvidence && (
                <div className="ai-evidence">
                  <strong>Evidence used</strong>
                  <ul>
                    <li>Database state: {aiEvidence.event ? 'event loaded' : aiEvidence.experiment ? 'experiment loaded' : 'not available'}</li>
                    <li>Counterfactual replay: {aiEvidence.decisions ? 'available' : 'not available'}</li>
                    <li>Provenance: {aiEvidence.provenance ? 'available' : 'not available'}</li>
                    <li>Recovery: {aiEvidence.recovery ? 'available' : 'not available'}</li>
                    <li>Verification: {aiEvidence.verification ? 'available' : 'not available'}</li>
                    <li>
                      Universal Logs: {aiEvidence.universal_log_context.returned_event_count} of {aiEvidence.universal_log_context.event_count} events
                      {aiEvidence.universal_log_context.truncated ? ' (partial context)' : ''}
                    </li>
                  </ul>
                  {aiEvidence.universal_log_context.events.length > 0 && (
                    <div className="ai-log-evidence">
                      {aiEvidence.universal_log_context.events.slice(-6).map((logEvent) => {
                        const logMsg = logEvent.details.message;
                        const operation = logEvent.details.operation;
                        const correlationId = logEvent.details.correlation_id;
                        return (
                          <div className="ai-log-item" key={logEvent.id}>
                            <small>
                              {formatDate(logEvent.timestamp)} · {logEvent.event_type}
                              {typeof operation === 'string' ? ` · ${operation}` : ''}
                              {typeof correlationId === 'string' ? ` · ${correlationId}` : ''}
                            </small>
                            <span>{typeof logMsg === 'string' ? logMsg : logEvent.source}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              <div className="view-navigation-footer">
                <button className="secondary" onClick={() => setWorkbenchView('rewind')}>
                  ← BACK TO CORRECTION &amp; PROVENANCE
                </button>
                <button onClick={() => setWorkbenchView('audit')}>
                  VIEW AUDIT TRAIL <span>→</span>
                </button>
              </div>
            </section>
          </div>
        )}

        {/* =================================================================
            VIEW 05: AUDIT TRAIL
            ================================================================= */}
        {workbenchView === 'audit' && (
          <div className="view-stage view-stage-audit">
            <section className="panel audit-panel">
              <div className="audit-heading">
                <div>
                  <div className="eyebrow">STAGE 05 // PERSISTED AUDIT TRAIL</div>
                  <h2>Governance &amp; History</h2>
                  <p className="stage-description">
                    Tamper-proof chronological log of all experiment generations, training iterations, counterfactual previews, and rewind transactions.
                  </p>
                </div>
                <button className="secondary audit-download" onClick={downloadAudit} disabled={!dataset || !audit.length}>
                  DOWNLOAD JSON EXPORT ↗
                </button>
              </div>

              <div className="timeline">
                {audit.length ? audit.slice(0, 16).map((entry) => (
                  <div className="timeline-item" key={entry.audit_id}>
                    <div className="audit-record-header">
                      <strong>{auditLabel(entry.operation)}</strong>
                      <time dateTime={entry.timestamp}>{formatDate(entry.timestamp)}</time>
                    </div>
                    <span className="audit-event-id">{entry.event_id || dataset?.dataset_id}</span>
                    <div className="audit-record-details"><AuditValue value={entry.details} /></div>
                  </div>
                )) : <p>No audit records for this dataset.</p>}
                {audit.length > 16 && <p className="audit-count">Showing 16 most recent of {audit.length} records. Download JSON for the full audit trail.</p>}
              </div>

              <div className="view-navigation-footer">
                <button className="secondary" onClick={() => setWorkbenchView('ai')}>
                  ← BACK TO AI REASONING
                </button>
                <button onClick={() => setWorkbenchView('records')}>
                  RETURN TO DATASET EXPLORER <span>→</span>
                </button>
              </div>
            </section>
          </div>
        )}
      </main>
      {showGenerateDialog && (
        <div className="generate-modal-backdrop" role="presentation">
          <section className="generate-modal" role="dialog" aria-modal="true" aria-labelledby="generate-modal-title-workbench">
            <div className="generate-modal-mark" aria-hidden="true">↗</div>
            <p className="eyebrow">NEW EXPERIMENT // CONFIRM</p>
            <h2 id="generate-modal-title-workbench">Create a fresh<br /><span>200-record experiment?</span></h2>
            <p>
              A new unlabeled experiment will be generated from the frozen global model.
              Your current dataset will remain stored for reproducibility.
            </p>
            <div className="generate-modal-actions">
              <button className="secondary" onClick={() => setShowGenerateDialog(false)}>CANCEL</button>
              <button onClick={() => void generateDataset()}>GENERATE EXPERIMENT <span>↗</span></button>
            </div>
          </section>
        </div>
      )}
      </div>
    )}
  </div>
  );
}
