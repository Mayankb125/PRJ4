import React, { useState } from 'react';
import { PredictionRecord } from '../types';
import { Brain, Flame, ShieldCheck, Code2, ChevronDown, ChevronUp, Clock, Sparkles } from 'lucide-react';

interface StressCardProps {
  prediction?: Partial<PredictionRecord>;
  hasBaseline: boolean;
  baselineBpm?: number | null;
  currentWindowElapsed: number;
  windowDurationS: number;
  completedWindows: number;
  latestFeatures?: Record<string, any>;
  baselineFeatures?: Record<string, number> | null;
}

export const StressCard: React.FC<StressCardProps> = ({
  prediction,
  hasBaseline,
  currentWindowElapsed,
  windowDurationS,
  completedWindows,
  latestFeatures,
  baselineFeatures,
}) => {
  const [showJson, setShowJson] = useState(false);
  const [showDelta, setShowDelta] = useState(false);

  const isStress = prediction?.vr_state === 'High' || prediction?.predicted_label === 'stress';
  const confidence = prediction?.confidence ?? 0;
  const stressProb = prediction?.stress_probability ?? 0;
  const rfProb = prediction?.rf_stress_probability ?? 0;
  const deltaBpm = prediction?.delta_bpm ?? 0;
  const decisionSource = prediction?.decision_source ?? 'model';
  const fourClass = prediction?.four_class_label ?? 'baseline';

  const progressPercent = Math.min(
    100,
    Math.round((currentWindowElapsed / (windowDurationS || 30)) * 100)
  );

  const vrPayload = {
    type: 'reading',
    window_index: prediction?.window_index ?? 0,
    status: 'Valid',
    state: prediction?.vr_state ?? 'Low',
    confidence: Number(confidence.toFixed(3)),
    stress_prob: Number(stressProb.toFixed(3)),
    bpm: prediction?.calculated_bpm ?? 0,
    model: 'rsp_only_rf_5feat_v1',
  };

  const featureNames = [
    { key: 'rsp_mean_amplitude', label: 'Mean Amplitude (N)' },
    { key: 'rsp_std_amplitude', label: 'Std Amplitude (N)' },
    { key: 'rsp_spectral_entropy', label: 'Spectral Entropy' },
    { key: 'rsp_mean_peak_valley_magnitude', label: 'Mean Depth (N)' },
    { key: 'rsp_std_peak_valley_magnitude', label: 'Std Depth (N)' },
  ];

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-full">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-200 mb-5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center shadow-xs">
              <Brain className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold font-display tracking-tight text-slate-900">
                Respiration Stress Classifier
              </h2>
              <div className="text-xs text-slate-500 font-medium">Bian et al. Random Forest · 81.06% LOSO Accuracy</div>
            </div>
          </div>
          <span className="text-xs font-semibold px-3 py-1.5 rounded-lg bg-indigo-50 text-indigo-700 border border-indigo-200">
            Window #{completedWindows}
          </span>
        </div>

        {/* Hero State Banner */}
        <div
          className={`rounded-2xl p-5 mb-5 border flex items-center justify-between transition-all ${
            !hasBaseline
              ? 'bg-amber-50/80 border-amber-200 text-amber-900'
              : isStress
              ? 'bg-rose-50/90 border-rose-200 text-rose-900'
              : 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
          }`}
        >
          <div className="flex items-center gap-4">
            <div className="shrink-0">
              {!hasBaseline ? (
                <div className="w-12 h-12 rounded-xl bg-amber-100 flex items-center justify-center">
                  <Clock className="w-6 h-6 text-amber-700 animate-pulse" />
                </div>
              ) : isStress ? (
                <div className="w-12 h-12 rounded-xl bg-rose-100 flex items-center justify-center">
                  <Flame className="w-6 h-6 text-rose-600 animate-bounce" />
                </div>
              ) : (
                <div className="w-12 h-12 rounded-xl bg-emerald-100 flex items-center justify-center">
                  <ShieldCheck className="w-6 h-6 text-emerald-700" />
                </div>
              )}
            </div>

            <div>
              <div className="text-xs font-bold uppercase tracking-wider opacity-70">
                Current Physiological State
              </div>
              <div className="text-xl sm:text-2xl font-black font-display tracking-tight mt-0.5">
                {!hasBaseline
                  ? 'Calibrating Baseline'
                  : isStress
                  ? 'High Stress Detected'
                  : 'Calm / Resting Baseline'}
              </div>
              <div className="text-xs font-medium opacity-80 mt-1">
                {!hasBaseline
                  ? `Recording baseline window (${currentWindowElapsed}s / ${windowDurationS}s)...`
                  : isStress
                  ? 'Breathing rate elevation and amplitude dynamics exceed stress boundary.'
                  : 'Breathing pattern is calm and within resting threshold limits.'}
              </div>
            </div>
          </div>

          <div className="text-right hidden sm:block">
            <div className="text-xs font-semibold opacity-70 uppercase tracking-wider">Confidence</div>
            <div className="text-2xl font-black font-mono">
              {(confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Stepped Stress Meter */}
        <div className="mb-5 bg-slate-50 border border-slate-200/90 p-4 rounded-xl">
          <div className="flex justify-between items-center text-xs font-semibold text-slate-700 mb-2">
            <span className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
              Model Stress Probability
            </span>
            <span className="font-mono font-bold text-sm text-slate-900">
              {(stressProb * 100).toFixed(1)}%
            </span>
          </div>

          {/* Meter Track */}
          <div className="relative w-full h-3.5 bg-slate-200 rounded-full overflow-hidden shadow-inner">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                isStress
                  ? 'bg-gradient-to-r from-amber-400 to-rose-500'
                  : 'bg-gradient-to-r from-blue-400 to-emerald-500'
              }`}
              style={{ width: `${Math.max(4, stressProb * 100)}%` }}
            />
          </div>

          <div className="flex justify-between text-[11px] text-slate-400 font-medium mt-1.5">
            <span>0% (Calm)</span>
            <span className="font-semibold text-slate-500">Threshold: 50%</span>
            <span>100% (High Stress)</span>
          </div>
        </div>

        {/* Diagnostic Stat Badges */}
        <div className="grid grid-cols-2 gap-3.5 mb-4">
          <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Rate Elevation (ΔBPM)</div>
            <div className="text-xl font-bold font-mono text-slate-900 mt-1">
              {deltaBpm >= 0 ? `+${deltaBpm.toFixed(1)}` : deltaBpm.toFixed(1)} <span className="text-xs font-normal text-slate-400">BPM</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              vs personal resting baseline
            </div>
          </div>

          <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Classification Source</div>
            <div className="text-base font-bold text-slate-900 mt-1 capitalize">
              {decisionSource === 'model' ? 'Random Forest' : 'Rate Threshold'}
            </div>
            <div className="text-[11px] text-slate-500 font-mono mt-0.5">
              RF: {(rfProb * 100).toFixed(0)}% · Class: {fourClass}
            </div>
          </div>
        </div>

        {/* Next Window Countdown Bar */}
        <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 mb-4">
          <div className="flex justify-between items-center text-xs text-slate-600 mb-1.5">
            <span className="font-medium">Window Progress (30s Step)</span>
            <span className="font-mono font-bold text-slate-800">
              {currentWindowElapsed}s / {windowDurationS}s
            </span>
          </div>
          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Collapsible Drawers: Features & Unity Payload */}
      <div className="pt-3 border-t border-slate-200 space-y-2">
        {/* Toggle 5 Features Drawer */}
        <button
          onClick={() => setShowDelta(!showDelta)}
          className="w-full flex items-center justify-between text-xs font-semibold text-slate-600 hover:text-slate-900 py-1.5 transition"
        >
          <span>5-Feature Values & Normalized Baseline Deltas</span>
          {showDelta ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showDelta && (
          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs animate-fadeIn">
            <table className="w-full text-left">
              <thead>
                <tr className="text-slate-400 border-b border-slate-200/80 pb-1 font-semibold uppercase text-[10px]">
                  <th className="py-1">Feature</th>
                  <th className="py-1 text-right">Live</th>
                  <th className="py-1 text-right">Baseline</th>
                  <th className="py-1 text-right">Delta (Δ)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                {featureNames.map((f) => {
                  const live = latestFeatures ? Number(latestFeatures[f.key] ?? 0) : 0;
                  const base = baselineFeatures ? Number(baselineFeatures[f.key] ?? 0) : 0;
                  const delta = live - base;
                  return (
                    <tr key={f.key} className="py-1">
                      <td className="py-1 font-sans text-slate-800 font-medium">{f.label}</td>
                      <td className="py-1 text-right">{live.toFixed(3)}</td>
                      <td className="py-1 text-right text-slate-500">{base.toFixed(3)}</td>
                      <td className={`py-1 text-right font-bold ${delta >= 0 ? 'text-blue-700' : 'text-slate-600'}`}>
                        {delta >= 0 ? `+${delta.toFixed(3)}` : delta.toFixed(3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Toggle Quest VR TCP JSON Payload */}
        <button
          onClick={() => setShowJson(!showJson)}
          className="w-full flex items-center justify-between text-xs font-semibold text-slate-600 hover:text-slate-900 py-1.5 transition"
        >
          <span className="flex items-center gap-1.5">
            <Code2 className="w-3.5 h-3.5 text-indigo-600" /> Quest VR TCP Packet Preview (:5005)
          </span>
          {showJson ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showJson && (
          <pre className="p-3 bg-slate-900 text-slate-100 rounded-xl text-[11px] font-mono overflow-x-auto shadow-inner animate-fadeIn">
            {JSON.stringify(vrPayload, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
};
