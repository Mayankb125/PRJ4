import React, { useState, useEffect } from 'react';
import { SystemStatus, BaselineCandidate } from '../types';
import {
  Play,
  Square,
  RefreshCw,
  Battery,
  Timer,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  SlidersHorizontal,
  Bluetooth,
  AlertCircle,
  FolderOpen,
  Trash2,
} from 'lucide-react';
import {
  startRecording,
  stopRecording,
  triggerRebaseline,
  fetchBaselineCandidates,
  loadBaselineCsv,
  clearBaseline,
} from '../api';

interface ControlBarProps {
  status: SystemStatus;
  onRefresh: () => void;
}

const TIMING_PATTERNS = [
  {
    label: '60s Baseline → 60s Rolling / 30s Updates (Quest VR Recommended)',
    base: 60,
    roll: 60,
    step: 30,
  },
  {
    label: '120s Baseline → 60s Rolling / 30s Updates',
    base: 120,
    roll: 60,
    step: 30,
  },
  {
    label: '120s Baseline → 30s Non-Overlapping Updates',
    base: 120,
    roll: 30,
    step: 30,
  },
  {
    label: '120s Fixed (2-Minute Windows)',
    base: 120,
    roll: 120,
    step: 120,
  },
  {
    label: '180s Fixed (3-Minute Windows — Standard WESAD)',
    base: 180,
    roll: 180,
    step: 180,
  },
  {
    label: '30s Fixed (Quick Live Test)',
    base: 30,
    roll: 30,
    step: 30,
  },
];

export const ControlBar: React.FC<ControlBarProps> = ({ status, onRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Calibration Drawer and Settings
  const [showSettings, setShowSettings] = useState(false);
  const [selectedHz, setSelectedHz] = useState<number>(20);
  const [selectedPatternIdx, setSelectedPatternIdx] = useState<number>(0);
  const [candidates, setCandidates] = useState<BaselineCandidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<string>('');

  useEffect(() => {
    fetchBaselineCandidates()
      .then((res) => {
        setCandidates(res.candidates);
        if (res.candidates.length > 0) {
          setSelectedCandidate(res.candidates[0].filename);
        }
      })
      .catch(() => {});
  }, []);

  const handleStart = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      const pat = TIMING_PATTERNS[selectedPatternIdx];
      await startRecording({
        sample_rate_hz: selectedHz,
        baseline_duration_s: pat.base,
        rolling_window_s: pat.roll,
        prediction_step_s: pat.step,
      });
      setTimeout(onRefresh, 600);
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to start acquisition');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      await stopRecording();
      setTimeout(onRefresh, 600);
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to stop acquisition');
    } finally {
      setLoading(false);
    }
  };

  const handleRebaseline = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      await triggerRebaseline();
      setTimeout(onRefresh, 600);
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to trigger rebaseline');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadBaseline = async () => {
    if (!selectedCandidate) return;
    try {
      setLoading(true);
      setErrorMsg(null);
      await loadBaselineCsv(selectedCandidate, 60.0);
      onRefresh();
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to pre-load baseline CSV');
    } finally {
      setLoading(false);
    }
  };

  const handleClearBaseline = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      await clearBaseline();
      onRefresh();
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to clear baseline');
    } finally {
      setLoading(false);
    }
  };

  const formatTimer = (totalSeconds: number) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const currentPattern = TIMING_PATTERNS[selectedPatternIdx];

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-5 mb-6 shadow-sm">
      {/* Top Main Status Bar */}
      <div className="flex flex-wrap items-center justify-between gap-5">
        {/* Left: Device Telemetry & Timing Status */}
        <div className="flex flex-wrap items-center gap-6 sm:gap-8">
          {/* Connection Status Indicator */}
          <div className="flex items-center gap-3.5">
            <div className="relative flex items-center justify-center">
              {status.is_running ? (
                <>
                  <span className="animate-ping absolute inline-flex h-4 w-4 rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-emerald-600"></span>
                </>
              ) : status.is_connecting ? (
                <>
                  <span className="animate-ping absolute inline-flex h-4 w-4 rounded-full bg-amber-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-amber-500"></span>
                </>
              ) : (
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-slate-300"></span>
              )}
            </div>

            <div>
              <div className="text-sm font-bold tracking-tight">
                {status.is_running ? (
                  <span className="text-emerald-700 flex items-center gap-1.5">
                    Streaming Active
                  </span>
                ) : status.is_connecting ? (
                  <span className="text-amber-700 flex items-center gap-1.5">
                    Searching Bluetooth...
                  </span>
                ) : (
                  <span className="text-slate-700 flex items-center gap-1.5">
                    Standby / Ready
                  </span>
                )}
              </div>
              <div className="text-xs text-slate-500 font-medium">
                {status.device_name || 'Vernier Respiration Belt (GDX-RB)'}
              </div>
            </div>
          </div>

          {/* Battery Level */}
          <div className="flex items-center gap-3 border-l border-slate-200/80 pl-6">
            <div className="w-9 h-9 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center">
              <Battery className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Battery</div>
              <div className="text-base font-bold font-mono text-slate-800">
                {status.battery_level !== null ? `${status.battery_level}%` : '—'}
              </div>
            </div>
          </div>

          {/* Session Timer */}
          <div className="flex items-center gap-3 border-l border-slate-200/80 pl-6">
            <div className="w-9 h-9 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center">
              <Timer className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Session Time</div>
              <div className="text-base font-bold font-mono text-slate-900 tracking-wide">
                {formatTimer(status.elapsed_seconds)}
              </div>
            </div>
          </div>

          {/* Current Active Pattern Pill */}
          <div className="hidden xl:flex items-center gap-2 border-l border-slate-200/80 pl-6">
            <SlidersHorizontal className="w-4 h-4 text-blue-600" />
            <div className="text-xs text-slate-600">
              <span className="font-semibold text-blue-700">{selectedHz} Hz</span> ·{' '}
              <span className="font-semibold text-slate-800">{currentPattern.base}s Baseline</span> →{' '}
              <span>{currentPattern.roll}s Rolling / {currentPattern.step}s Tick</span>
            </div>
          </div>
        </div>

        {/* Right: Primary Action Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 px-3.5 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-sm font-semibold border border-slate-300/80 transition cursor-pointer"
          >
            {status.has_baseline ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            ) : (
              <SlidersHorizontal className="w-4 h-4 text-slate-500" />
            )}
            <span>Settings & Baseline</span>
            {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          <button
            onClick={handleRebaseline}
            disabled={loading || !status.is_running}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold border transition ${
              !status.is_running
                ? 'opacity-40 cursor-not-allowed bg-slate-100 text-slate-400 border-slate-200'
                : 'bg-white hover:bg-slate-50 text-slate-800 border-slate-300 shadow-xs cursor-pointer'
            }`}
            title="Clear baseline and re-calibrate fresh baseline from current respiration"
          >
            <RefreshCw className={`w-4 h-4 text-blue-600 ${loading ? 'animate-spin' : ''}`} />
            <span>Re-Calibrate</span>
          </button>

          {!status.is_running ? (
            <button
              onClick={handleStart}
              disabled={loading || status.is_connecting}
              className={`flex items-center gap-2.5 px-6 py-2.5 rounded-xl text-sm font-bold text-white shadow-md transition cursor-pointer ${
                status.is_connecting
                  ? 'bg-amber-600 hover:bg-amber-700 shadow-amber-600/20'
                  : 'bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/25 active:scale-[0.98]'
              }`}
            >
              {status.is_connecting ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Searching Belt...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Start Recording</span>
                </>
              )}
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={loading}
              className="flex items-center gap-2.5 px-6 py-2.5 bg-rose-600 hover:bg-rose-700 text-white rounded-xl text-sm font-bold shadow-md shadow-rose-600/25 transition cursor-pointer active:scale-[0.98]"
            >
              <Square className="w-4 h-4 fill-white" />
              <span>Stop Recording</span>
            </button>
          )}
        </div>
      </div>

      {/* Helpful Sensor Search Notice */}
      {status.is_connecting && (
        <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-3 text-sm text-amber-900 animate-fadeIn">
          <Bluetooth className="w-5 h-5 text-amber-600 shrink-0 mt-0.5 animate-pulse" />
          <div>
            <div className="font-bold">Connecting to Vernier Go Direct Respiration Belt...</div>
            <div className="text-xs text-amber-800 mt-1">
              Ensure your belt sensor is powered on (LED blinking green) and positioned near this computer.
              Acquisition will start automatically as soon as Bluetooth syncs.
            </div>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="mt-4 p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-3 text-sm text-rose-800">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
          <span><b>Error:</b> {errorMsg}</span>
        </div>
      )}

      {/* Expandable Baseline & Settings Panel */}
      {showSettings && (
        <div className="mt-5 pt-5 border-t border-slate-200 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Column 1: Timing Pattern */}
          <div className="space-y-3">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              1. Calibration & Inference Window
            </label>
            <select
              value={selectedPatternIdx}
              onChange={(e) => setSelectedPatternIdx(Number(e.target.value))}
              disabled={status.is_running}
              className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {TIMING_PATTERNS.map((p, idx) => (
                <option key={idx} value={idx}>
                  {p.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500">
              Quest VR runs with a 60s baseline, then evaluates a 60s rolling window every 30s.
            </p>
          </div>

          {/* Column 2: Sampling Rate */}
          <div className="space-y-3">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              2. Sampling Frequency
            </label>
            <div className="flex gap-2">
              {[10, 20, 50].map((hz) => (
                <button
                  key={hz}
                  type="button"
                  onClick={() => setSelectedHz(hz)}
                  disabled={status.is_running}
                  className={`flex-1 py-2 text-xs font-bold rounded-lg border transition ${
                    selectedHz === hz
                      ? 'bg-blue-600 text-white border-blue-600 shadow-xs'
                      : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100 disabled:opacity-50'
                  }`}
                >
                  {hz} Hz
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              Default is 20 Hz, matching WESAD Respiration chest belt ground truth.
            </p>
          </div>

          {/* Column 3: Pre-load Calm Baseline CSV */}
          <div className="space-y-3">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              3. Fast Calm Baseline (Optional)
            </label>
            <div className="flex gap-2">
              <select
                value={selectedCandidate}
                onChange={(e) => setSelectedCandidate(e.target.value)}
                className="flex-1 text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 text-slate-800 truncate"
              >
                {candidates.length === 0 ? (
                  <option value="">No past session CSVs found</option>
                ) : (
                  candidates.map((c) => (
                    <option key={c.filename} value={c.filename}>
                      {c.filename} ({c.size_kb} KB)
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                onClick={handleLoadBaseline}
                disabled={loading || candidates.length === 0}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-lg transition disabled:opacity-50 flex items-center gap-1 shadow-xs"
                title="Load 60s calm respiration from this recording immediately"
              >
                <FolderOpen className="w-3.5 h-3.5" />
                <span>Load</span>
              </button>
              {status.has_baseline && (
                <button
                  type="button"
                  onClick={handleClearBaseline}
                  disabled={loading}
                  className="px-2.5 py-2 bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 text-xs font-bold rounded-lg transition"
                  title="Clear baseline"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
            <p className="text-xs text-slate-500">
              Skip live calibration by importing a validated calm sitting baseline.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
