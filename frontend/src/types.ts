export interface WaveformPoint {
  elapsed: number;
  force: number;
  bpm?: number | null;
}

export interface PredictionRecord {
  timestamp_utc: string;
  window_index: number;
  window_start_seconds: number;
  window_end_seconds: number;
  is_baseline_window: boolean;
  calculated_bpm: number;
  baseline_bpm: number;
  delta_bpm: number;
  n_breaths: number;
  predicted_label: string;
  vr_state: string;
  confidence: number;
  stress_probability: number;
  rf_stress_probability: number;
  rate_stress_probability: number;
  decision_source: string;
  baseline_probability: number;
  four_class_label: string;
}

export interface SystemStatus {
  is_running: boolean;
  is_connecting: boolean;
  connected: boolean;
  error_message: string | null;
  device_name: string;
  battery_level: number | null;
  sample_rate_hz: number;
  window_duration_s: number;
  baseline_duration_s: number;
  rolling_window_s: number;
  prediction_step_s: number;
  sample_count: number;
  latest_force: number;
  latest_bpm: number | null;
  elapsed_seconds: number;
  completed_windows: number;
  current_window_elapsed: number;
  unity_link: {
    status: string;
    clients: number;
    last_error: string | null;
  };
  has_baseline: boolean;
  baseline_bpm: number | null;
  baseline_source: string | null;
  baseline_warning: string | null;
  baseline_features: Record<string, number> | null;
  latest_features: Record<string, any>;
  latest_prediction: Partial<PredictionRecord>;
  prediction_history: PredictionRecord[];
  recent_waveform: WaveformPoint[];
  raw_file?: string | null;
  predictions_file?: string | null;
  sessions_dir?: string | null;
  live_data_dir?: string | null;
}

export interface UnityTrial {
  global_trial: number;
  block_trial: number;
  prev_level: number;
  next_level: number;
  prev_guidance: number;
  next_guidance: number;
  change: string;
  perf_class: string;
  physio_state: string;
  physio_confidence: number;
  rule_cell: string;
  reason: string;
  start_ist: string;
  end_ist: string;
  outcome: string;
}

export interface UnityEvent {
  ist: string;
  type: string;
  detail: string;
  session_s: number;
}

export interface UnityLatest {
  session_name: string | null;
  session_path?: string | null;
  unity_root?: string;
  source_mode?: string;
  trials: UnityTrial[];
  latest_trial: UnityTrial | null;
  recent_events: UnityEvent[];
  rules: Record<string, string>;
  cell_counts: Record<string, number>;
  active_cell: string | null;
  hold_no_physio: boolean;
}

export interface BaselineCandidate {
  filename: string;
  path: string;
  size_kb: number;
  modified: string;
}

export interface SourcesInfo {
  source_mode: 'quest' | 'pc';
  unity_root: string;
  default_quest_root: string;
  default_pc_root: string;
  follow_newest: boolean;
  selected_session: string | null;
  available_sessions: string[];
  selected_prediction_file: string;
  available_prediction_files: string[];
  show_practice: boolean;
  storage_paths: {
    live_sessions_dir: string;
    baseline_candidates_dir: string;
    active_raw_file: string | null;
    active_predictions_file: string | null;
    active_unity_session_dir: string | null;
  };
}
