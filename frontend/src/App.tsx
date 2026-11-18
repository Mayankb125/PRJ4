import React, { useState, useEffect, useCallback } from 'react';
import { SystemStatus, UnityLatest } from './types';
import { fetchStatus, fetchUnityLatest } from './api';
import { ControlBar } from './components/ControlBar';
import { SourcesPanel } from './components/SourcesPanel';
import { WaveformChart } from './components/WaveformChart';
import { StressCard } from './components/StressCard';
import { VRAdaptivePanel } from './components/VRAdaptivePanel';
import { TrialsTable } from './components/TrialsTable';
import { Activity, ShieldAlert, Cpu, HardDrive, ChevronDown, ChevronUp, FolderCheck } from 'lucide-react';

export const App: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [unityData, setUnityData] = useState<UnityLatest>({
    session_name: null,
    trials: [],
    latest_trial: null,
    recent_events: [],
    rules: {},
    cell_counts: {},
    active_cell: null,
    hold_no_physio: false,
  });
  const [showSources, setShowSources] = useState<boolean>(false);
  const [apiOnline, setApiOnline] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);

  // Poll status from backend bridge
  const loadData = useCallback(async () => {
    try {
      const snap = await fetchStatus();
      setStatus(snap);
      setApiOnline(true);
      setApiError(null);
    } catch (err: any) {
      setApiOnline(false);
      setApiError(err.message);
    }

    try {
      const udata = await fetchUnityLatest();
      setUnityData(udata);
    } catch {
      // Unity session poll failure is non-fatal
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 1000);
    return () => clearInterval(interval);
  }, [loadData]);

  if (!status) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center text-slate-700">
        <Activity className="w-12 h-12 text-blue-600 animate-spin mb-4" />
        <div className="text-lg font-bold text-slate-800">
          Connecting to Biomedical Telemetry Server...
        </div>
        <div className="text-sm text-slate-500 font-mono mt-1">
          http://127.0.0.1:8000
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50/80 text-slate-800 p-4 sm:p-6 lg:p-8 flex flex-col justify-between max-w-[1840px] mx-auto">
      {/* Top Header */}
      <div>
        <header className="flex flex-wrap items-center justify-between gap-4 pb-5 mb-5 border-b border-slate-200">
          <div className="flex items-center gap-3.5">
            <div className="w-12 h-12 rounded-xl bg-blue-600 text-white flex items-center justify-center shadow-md shadow-blue-500/20">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold font-display tracking-tight text-slate-900 flex items-center gap-2">
                Physiological Affective Computing Workstation
              </h1>
              <div className="text-sm text-slate-500 font-medium">
                Vernier Go Direct Respiration Belt + Closed-Loop Adaptive VR Maze
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm">
            {/* Sources & Storage Inspector Button */}
            <button
              onClick={() => setShowSources(!showSources)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-sm font-bold transition shadow-xs cursor-pointer ${
                showSources
                  ? 'bg-blue-600 text-white border-blue-600 shadow-blue-500/20'
                  : 'bg-white hover:bg-slate-50 text-slate-700 border-slate-200'
              }`}
              title="Inspect data storage folders and configure Unity VR sources"
            >
              <HardDrive className="w-4 h-4" />
              <span>Sources & Storage</span>
              {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            <div className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-white border border-slate-200 shadow-xs text-slate-700 font-medium">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  apiOnline ? 'bg-emerald-500 shadow-xs shadow-emerald-500/50' : 'bg-rose-500'
                }`}
              />
              <span>API {apiOnline ? 'Online (:8000)' : 'Disconnected'}</span>
            </div>

            <div className="hidden sm:flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 shadow-xs text-sm">
              <span className="text-slate-400 font-medium">Model:</span>
              <span className="text-blue-700 font-semibold">RSP-Only RF (81.06% LOSO)</span>
            </div>
          </div>
        </header>

        {apiError && (
          <div className="mb-5 p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-3 text-sm text-rose-800 shadow-xs">
            <ShieldAlert className="w-5 h-5 text-rose-600 shrink-0" />
            <span><b>Bridge Connection Error:</b> {apiError}. Ensure the Python FastAPI server is running on port 8000.</span>
          </div>
        )}

        {/* Sources & Data Storage Inspector Drawer */}
        {showSources && (
          <SourcesPanel status={status} onRefresh={loadData} />
        )}

        {/* Global Control Bar */}
        <ControlBar status={status} onRefresh={loadData} />

        {/* Storage Quick Path Banner (Always shows where files are being stored) */}
        <div className="mb-6 px-4 py-2.5 bg-white border border-slate-200/90 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs shadow-2xs">
          <div className="flex items-center gap-2 text-slate-600">
            <FolderCheck className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>
              <b>Active Data Storage:</b> <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded font-mono">D:\PRJ\RSP_Only_Model\live_sessions\</code>
              {status.raw_file && (
                <span className="ml-2 text-emerald-700 font-medium font-mono truncate max-w-md hidden md:inline">
                  (Writing: {status.raw_file.split('\\').pop()})
                </span>
              )}
            </span>
          </div>

          <button
            onClick={() => setShowSources(true)}
            className="text-blue-600 hover:text-blue-800 font-semibold underline cursor-pointer"
          >
            Configure Sources & View Full Paths →
          </button>
        </div>

        {/* Row 1: Physiological Signal & ML Inference (Side-by-Side 50/50) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Left: Live Respiration Strip Chart & Rate Trajectory */}
          <div className="h-full">
            <WaveformChart
              data={status.recent_waveform || []}
              latestForce={status.latest_force}
              latestBpm={status.latest_bpm}
              baselineBpm={status.baseline_bpm}
              sampleRateHz={status.sample_rate_hz}
              sampleCount={status.sample_count}
              isRunning={status.is_running}
              predictionHistory={status.prediction_history || []}
            />
          </div>

          {/* Right: 5-Feature Respiration ML Stress Inference */}
          <div className="h-full">
            <StressCard
              prediction={status.latest_prediction}
              hasBaseline={status.has_baseline}
              baselineBpm={status.baseline_bpm}
              currentWindowElapsed={status.current_window_elapsed}
              windowDurationS={status.window_duration_s}
              completedWindows={status.completed_windows}
              latestFeatures={status.latest_features}
              baselineFeatures={status.baseline_features}
            />
          </div>
        </div>

        {/* Row 2: VR Headset Adaptive Bridge (Downward) */}
        <div className="mb-6">
          <VRAdaptivePanel
            unityData={unityData}
            unityLink={status.unity_link}
          />
        </div>

        {/* Row 3: Trial Decisions Audit Table */}
        <div className="mb-6">
          <TrialsTable trials={unityData?.trials || []} />
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-8 pt-4 border-t border-slate-200 text-center text-xs text-slate-500 flex flex-wrap items-center justify-between gap-3">
        <div>PRJ4 Affective Computing Laboratory · Bian et al. (2019) & WESAD Multimodal Protocol</div>
        <div>Storage: <code>D:\PRJ\RSP_Only_Model\live_sessions\</code> · Quest TCP Port 5005</div>
      </footer>
    </div>
  );
};
