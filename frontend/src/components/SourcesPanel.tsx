import React, { useState, useEffect } from 'react';
import { SourcesInfo, SystemStatus } from '../types';
import { fetchSources, updateSources } from '../api';
import {
  HardDrive,
  FolderSync,
  FileSpreadsheet,
  Check,
  Copy,
  HelpCircle,
  Headphones,
  Monitor,
  Info,
} from 'lucide-react';

interface SourcesPanelProps {
  status: SystemStatus;
  onRefresh: () => void;
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({ status, onRefresh }) => {
  const [sources, setSources] = useState<SourcesInfo | null>(null);
  const [customRoot, setCustomRoot] = useState<string>('');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const loadSources = async () => {
    try {
      const data = await fetchSources();
      setSources(data);
      setCustomRoot(data.unity_root);
    } catch {
      // Non-fatal
    }
  };

  useEffect(() => {
    loadSources();
  }, []);

  const handleCopy = (key: string, text?: string | null) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const handleSourceModeChange = async (mode: 'quest' | 'pc') => {
    try {
      setSaving(true);
      await updateSources({ source_mode: mode, custom_root: '' });
      await loadSources();
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  const handleFollowChange = async (follow: boolean) => {
    try {
      setSaving(true);
      await updateSources({ follow_newest: follow });
      await loadSources();
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  const handleSessionChange = async (session: string) => {
    try {
      setSaving(true);
      await updateSources({ selected_session: session, follow_newest: false });
      await loadSources();
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  const handlePredFileChange = async (filename: string) => {
    try {
      setSaving(true);
      await updateSources({ selected_prediction_file: filename });
      await loadSources();
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  const handlePracticeChange = async (show: boolean) => {
    try {
      setSaving(true);
      await updateSources({ show_practice: show });
      await loadSources();
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  const handleRootSave = async () => {
    try {
      setSaving(true);
      await updateSources({ custom_root: customRoot });
      await loadSources();
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-white border-2 border-blue-200/90 rounded-2xl p-6 shadow-md mb-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between pb-4 border-b border-slate-200 mb-5 gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-center shadow-xs">
            <HardDrive className="w-5 h-5 text-blue-600" />
          </div>
          <div>
            <h2 className="text-lg font-bold font-display tracking-tight text-slate-900 flex items-center gap-2">
              Sources & Data Storage Inspector
            </h2>
            <div className="text-xs text-slate-500 font-medium">
              Configure Unity log paths, select sessions, and view exact hard drive storage locations
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs bg-blue-50 text-blue-700 px-3 py-1.5 rounded-lg border border-blue-200 font-semibold">
            Status: {saving ? 'Updating...' : 'Synchronized'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side (5 Cols): Sources Configuration (Replaces Streamlit Page 8 Sidebar) */}
        <div className="lg:col-span-5 bg-slate-50/90 border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700 border-b border-slate-200 pb-2">
            <FolderSync className="w-4 h-4 text-blue-600" />
            <span>Unity Sources Configuration</span>
          </div>

          {/* Unity is running on (Radio Buttons) */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
              <span>Unity is running on:</span>
              <span className="text-slate-400 font-normal" title="Select where the VR application is saving session logs">
                <HelpCircle className="w-3.5 h-3.5" />
              </span>
            </label>
            <div className="grid grid-cols-2 gap-2.5">
              <button
                type="button"
                onClick={() => handleSourceModeChange('pc')}
                className={`flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-bold border transition ${
                  sources?.source_mode === 'pc'
                    ? 'bg-blue-600 text-white border-blue-600 shadow-xs'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                }`}
              >
                <Monitor className="w-3.5 h-3.5" />
                <span>PC (editor / Link)</span>
              </button>

              <button
                type="button"
                onClick={() => handleSourceModeChange('quest')}
                className={`flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-bold border transition ${
                  sources?.source_mode === 'quest'
                    ? 'bg-blue-600 text-white border-blue-600 shadow-xs'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                }`}
              >
                <Headphones className="w-3.5 h-3.5" />
                <span>Quest standalone (USB)</span>
              </button>
            </div>
          </div>

          {/* Unity Log Folder Input */}
          <div className="space-y-1.5">
            <label className="text-xs font-bold text-slate-700 block">
              Unity Log Folder Path:
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={customRoot}
                onChange={(e) => setCustomRoot(e.target.value)}
                className="w-full text-xs font-mono bg-white border border-slate-300 rounded-lg p-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Path to Unity session folders..."
              />
              <button
                type="button"
                onClick={handleRootSave}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-900 text-white text-xs font-bold rounded-lg transition shrink-0"
              >
                Save
              </button>
            </div>
            <p className="text-[11px] text-slate-500 font-medium">
              {sources?.source_mode === 'quest'
                ? 'ADB pull destination folder: D:\\Vr_Project\\MazeStudy\\tools\\quest_usb\\pulled\\MazeStudy'
                : 'Windows LocalLow path: %USERPROFILE%\\AppData\\LocalLow\\DefaultCompany\\MazeStudy\\MazeStudy'}
            </p>
          </div>

          {/* Follow Newest Unity Session Checkbox */}
          <div className="pt-2 border-t border-slate-200">
            <label className="flex items-center gap-2.5 cursor-pointer text-xs font-semibold text-slate-800">
              <input
                type="checkbox"
                checked={sources?.follow_newest ?? true}
                onChange={(e) => handleFollowChange(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500 cursor-pointer"
              />
              <span>Follow newest Unity session automatically</span>
            </label>

            {!sources?.follow_newest && (
              <div className="mt-2.5 space-y-1">
                <label className="text-xs font-semibold text-slate-600">Select Specific Unity Session:</label>
                <select
                  value={sources?.selected_session || ''}
                  onChange={(e) => handleSessionChange(e.target.value)}
                  className="w-full text-xs font-mono bg-white border border-slate-300 rounded-lg p-2 text-slate-800"
                >
                  {sources?.available_sessions.length === 0 ? (
                    <option value="">No sessions found in folder</option>
                  ) : (
                    sources?.available_sessions.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))
                  )}
                </select>
              </div>
            )}
          </div>

          {/* ML Predictions File Dropdown */}
          <div className="pt-2 border-t border-slate-200 space-y-1.5">
            <label className="text-xs font-bold text-slate-700 block">
              ML Predictions File:
            </label>
            <select
              value={sources?.selected_prediction_file || '(current live session / newest)'}
              onChange={(e) => handlePredFileChange(e.target.value)}
              className="w-full text-xs font-mono bg-white border border-slate-300 rounded-lg p-2 text-slate-800"
            >
              {(sources?.available_prediction_files || ['(current live session / newest)']).map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>

          {/* Practice Trials Checkbox */}
          <div className="pt-2 border-t border-slate-200">
            <label className="flex items-center gap-2.5 cursor-pointer text-xs font-semibold text-slate-800">
              <input
                type="checkbox"
                checked={sources?.show_practice ?? false}
                onChange={(e) => handlePracticeChange(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500 cursor-pointer"
              />
              <span>Show practice trials in table & matrix</span>
            </label>
          </div>
        </div>

        {/* Right Side (7 Cols): Where is my data storing? (Explicit Path Inspector) */}
        <div className="lg:col-span-7 bg-slate-50/90 border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-2">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-700">
              <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
              <span>Exact Hard Drive Storage Locations</span>
            </div>
            <span className="text-[11px] text-slate-500 font-medium">Windows Drive D:\PRJ</span>
          </div>

          <div className="space-y-3">
            {/* Storage Item 1: Raw Respiration Force */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-3.5 shadow-2xs">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-blue-600"></span>
                  1. Live Respiration Force Waveform (CSV)
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy('raw', status.raw_file || sources?.storage_paths.live_sessions_dir)}
                  className="flex items-center gap-1 text-[11px] font-semibold text-blue-600 hover:text-blue-800 cursor-pointer"
                >
                  {copiedKey === 'raw' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedKey === 'raw' ? 'Copied' : 'Copy Path'}</span>
                </button>
              </div>

              <div className="font-mono text-xs text-slate-700 bg-slate-50 p-2 rounded border border-slate-200/80 truncate select-all">
                {status.raw_file || `${sources?.storage_paths.live_sessions_dir}\\raw_YYYYMMDD_HHMMSS_${status.sample_rate_hz}Hz.csv`}
              </div>
              <div className="flex justify-between items-center text-[11px] text-slate-500 mt-1">
                <span>Folder: <code>D:\PRJ\RSP_Only_Model\live_sessions\</code></span>
                <span>{status.is_running ? '🟢 Actively writing samples' : 'Standby'}</span>
              </div>
            </div>

            {/* Storage Item 2: ML Stress Predictions & Features */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-3.5 shadow-2xs">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
                  2. 5-Feature ML Predictions & Stress Classifications (CSV)
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy('pred', status.predictions_file || sources?.storage_paths.live_sessions_dir)}
                  className="flex items-center gap-1 text-[11px] font-semibold text-indigo-600 hover:text-indigo-800 cursor-pointer"
                >
                  {copiedKey === 'pred' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedKey === 'pred' ? 'Copied' : 'Copy Path'}</span>
                </button>
              </div>

              <div className="font-mono text-xs text-slate-700 bg-slate-50 p-2 rounded border border-slate-200/80 truncate select-all">
                {status.predictions_file || `${sources?.storage_paths.live_sessions_dir}\\predictions_YYYYMMDD_HHMMSS_${status.sample_rate_hz}Hz_*.csv`}
              </div>
              <div className="flex justify-between items-center text-[11px] text-slate-500 mt-1">
                <span>Features: <code>amp, std, entropy, depth_mean, depth_std</code></span>
                <span>{status.completed_windows} windows logged</span>
              </div>
            </div>

            {/* Storage Item 3: Meta Quest VR Headset Study Data */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-3.5 shadow-2xs">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
                  3. Quest VR Session Logs (trials.csv, events.csv, session.csv)
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy('unity', sources?.storage_paths.active_unity_session_dir || sources?.unity_root)}
                  className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600 hover:text-emerald-800 cursor-pointer"
                >
                  {copiedKey === 'unity' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedKey === 'unity' ? 'Copied' : 'Copy Path'}</span>
                </button>
              </div>

              <div className="font-mono text-xs text-slate-700 bg-slate-50 p-2 rounded border border-slate-200/80 truncate select-all">
                {sources?.storage_paths.active_unity_session_dir || sources?.unity_root}
              </div>
              <div className="flex justify-between items-center text-[11px] text-slate-500 mt-1">
                <span>Session: <b>{sources?.selected_session || 'No session folder yet'}</b></span>
                <span>Quest USB Bridge: <code>tools\quest_usb\start_quest_usb_bridge.bat</code></span>
              </div>
            </div>
          </div>

          <div className="p-3 bg-blue-50/80 border border-blue-200 rounded-xl text-xs text-blue-900 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
            <div>
              <b>Data Pipeline Summary:</b> When you click <b>Start Recording</b>, raw Vernier force samples are written directly to <code>RSP_Only_Model\live_sessions\raw_*.csv</code>. Every 30s prediction window, extracted features and model states are appended to <code>predictions_*.csv</code> and pushed over TCP port 5005 to Meta Quest. When Unity completes a maze trial, it writes to <code>trials.csv</code> and <code>events.csv</code>.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
