import { SystemStatus, UnityLatest, SourcesInfo, BaselineCandidate } from './types';

const API_BASE = '/api';

export async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch status: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchUnityLatest(): Promise<UnityLatest> {
  const res = await fetch(`${API_BASE}/unity/latest`);
  if (!res.ok) {
    throw new Error(`Failed to fetch Unity data: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSources(): Promise<SourcesInfo> {
  const res = await fetch(`${API_BASE}/sources`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sources: ${res.statusText}`);
  }
  return res.json();
}

export async function updateSources(payload: {
  source_mode?: string;
  custom_root?: string;
  follow_newest?: boolean;
  selected_session?: string;
  selected_prediction_file?: string;
  show_practice?: boolean;
}): Promise<any> {
  const res = await fetch(`${API_BASE}/sources`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Failed to update sources: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBaselineCandidates(): Promise<{ candidates: BaselineCandidate[] }> {
  const res = await fetch(`${API_BASE}/baseline/candidates`);
  if (!res.ok) {
    throw new Error('Failed to fetch baseline candidates');
  }
  return res.json();
}

export async function loadBaselineCsv(filename: string, windowSeconds = 60): Promise<any> {
  const res = await fetch(`${API_BASE}/baseline/load_csv`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, window_seconds: windowSeconds }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to load baseline CSV');
  }
  return res.json();
}

export async function clearBaseline(): Promise<any> {
  const res = await fetch(`${API_BASE}/baseline/clear`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error('Failed to clear baseline');
  }
  return res.json();
}

export async function startRecording(config?: {
  sample_rate_hz?: number;
  baseline_duration_s?: number;
  rolling_window_s?: number;
  prediction_step_s?: number;
}): Promise<any> {
  const res = await fetch(`${API_BASE}/control/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sample_rate_hz: config?.sample_rate_hz ?? 20,
      baseline_duration_s: config?.baseline_duration_s ?? 60,
      rolling_window_s: config?.rolling_window_s ?? 60,
      prediction_step_s: config?.prediction_step_s ?? 30,
      keep_baseline: false,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to start recording');
  }
  return res.json();
}

export async function stopRecording(): Promise<any> {
  const res = await fetch(`${API_BASE}/control/stop`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error('Failed to stop recording');
  }
  return res.json();
}

export async function triggerRebaseline(): Promise<any> {
  const res = await fetch(`${API_BASE}/control/rebaseline`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error('Failed to trigger rebaseline');
  }
  return res.json();
}
