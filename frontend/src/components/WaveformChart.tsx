import React, { useEffect, useRef } from 'react';
import { WaveformPoint, PredictionRecord } from '../types';
import { Activity, Gauge, TrendingUp } from 'lucide-react';

interface WaveformChartProps {
  data: WaveformPoint[];
  latestForce: number;
  latestBpm: number | null;
  baselineBpm: number | null;
  sampleRateHz: number;
  sampleCount: number;
  isRunning: boolean;
  predictionHistory?: PredictionRecord[];
}

export const WaveformChart: React.FC<WaveformChartProps> = ({
  data,
  latestForce,
  latestBpm,
  baselineBpm,
  sampleRateHz,
  sampleCount,
  isRunning,
  predictionHistory = [],
}) => {
  const forceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bpmCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // 1. Draw Force Oscilloscope Canvas (Clean Clinical Paper Style)
  useEffect(() => {
    const canvas = forceCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Background: Clean Pure White
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // Subtle Grid Lines
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#f1f5f9';
    for (let c = 1; c < 10; c++) {
      ctx.beginPath();
      ctx.moveTo((w / 10) * c, 0);
      ctx.lineTo((w / 10) * c, h);
      ctx.stroke();
    }
    for (let r = 1; r < 5; r++) {
      ctx.beginPath();
      ctx.moveTo(0, (h / 5) * r);
      ctx.lineTo(w, (h / 5) * r);
      ctx.stroke();
    }

    if (data.length < 2) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 13px "Outfit", system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        isRunning ? 'Acquiring 20 Hz respiration signal...' : 'Belt in standby — click "Start Recording"',
        w / 2,
        h / 2 + 5
      );
      return;
    }

    const forces = data.map((d) => d.force);
    let minForce = Math.min(...forces);
    let maxForce = Math.max(...forces);
    const range = Math.max(maxForce - minForce, 1.8);
    minForce -= range * 0.12;
    maxForce += range * 0.12;

    // Soft Blue Gradient Area Fill
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const normalizedY = (data[i].force - minForce) / (maxForce - minForce);
      const y = h - normalizedY * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(37, 99, 235, 0.08)';
    ctx.fill();

    // Vibrant Medical Royal Blue Waveform Trace
    ctx.beginPath();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = '#2563eb';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const normalizedY = (data[i].force - minForce) / (maxForce - minForce);
      const y = h - normalizedY * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Pulsing Lead Point
    if (data.length > 0) {
      const lastX = w;
      const lastY = h - ((data[data.length - 1].force - minForce) / (maxForce - minForce)) * h;

      ctx.beginPath();
      ctx.arc(lastX - 2, lastY, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#2563eb';
      ctx.fill();
    }
  }, [data, isRunning]);

  // 2. Draw Breathing Rate (BPM) Trajectory vs Baseline
  useEffect(() => {
    const canvas = bpmCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // Subtle Grid
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#f8fafc';
    for (let r = 1; r < 4; r++) {
      ctx.beginPath();
      ctx.moveTo(0, (h / 4) * r);
      ctx.lineTo(w, (h / 4) * r);
      ctx.stroke();
    }

    const minBpm = 8;
    const maxBpm = 32;

    const getY = (bpmVal: number) => {
      const clamped = Math.max(minBpm, Math.min(maxBpm, bpmVal));
      return h - ((clamped - minBpm) / (maxBpm - minBpm)) * (h - 24) - 12;
    };

    // Draw Personal Baseline Horizontal Reference
    if (baselineBpm !== null && baselineBpm > 0) {
      const baseY = getY(baselineBpm);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1.5;
      ctx.moveTo(0, baseY);
      ctx.lineTo(w, baseY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#64748b';
      ctx.font = '600 11px "Outfit", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Baseline: ${baselineBpm.toFixed(1)} BPM`, 8, baseY - 5);

      // +5.0 BPM Stress Rule Line (Bian et al.)
      const stressY = getY(baselineBpm + 5.0);
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.2;
      ctx.moveTo(0, stressY);
      ctx.lineTo(w, stressY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#e11d48';
      ctx.font = '600 11px "Outfit", sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`+5.0 BPM Stress Threshold`, w - 8, stressY - 4);
    }

    // Draw Rolling Window BPM History
    const points = predictionHistory.slice(-12);
    if (points.length >= 2) {
      ctx.beginPath();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#059669';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      points.forEach((p, idx) => {
        const x = (idx / (points.length - 1)) * (w - 36) + 18;
        const bpmVal = p.calculated_bpm || baselineBpm || 14;
        const y = getY(bpmVal);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Diamond marker on each prediction window
      points.forEach((p, idx) => {
        const x = (idx / (points.length - 1)) * (w - 36) + 18;
        const bpmVal = p.calculated_bpm || baselineBpm || 14;
        const y = getY(bpmVal);
        const isStress = p.vr_state === 'High' || p.predicted_label === 'stress';

        ctx.fillStyle = isStress ? '#e11d48' : '#059669';
        ctx.beginPath();
        ctx.moveTo(x, y - 5);
        ctx.lineTo(x + 5, y);
        ctx.lineTo(x, y + 5);
        ctx.lineTo(x - 5, y);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 11px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${bpmVal.toFixed(1)}`, x, y - 8);
      });
    } else {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '500 12px "Outfit", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Breathing rate trajectory updates with each 30s prediction window', w / 2, h / 2 + 4);
    }
  }, [baselineBpm, predictionHistory]);

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-full">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-200 mb-5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-center shadow-xs">
              <Activity className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold font-display tracking-tight text-slate-900">
                Respiration Waveform & Breathing Rate
              </h2>
              <div className="text-xs text-slate-500 font-medium">Vernier GDX-RB Telemetry & Frequency Analysis</div>
            </div>
          </div>
          <div className="flex items-center gap-2.5 text-xs">
            <span className="bg-slate-100 px-3 py-1.5 rounded-lg text-slate-700 font-medium border border-slate-200">
              Rate: <b className="text-blue-700 font-bold">{sampleRateHz} Hz</b>
            </span>
            <span className="bg-slate-100 px-3 py-1.5 rounded-lg text-slate-700 font-medium border border-slate-200">
              Samples: <b className="text-slate-900 font-bold font-mono">{sampleCount.toLocaleString()}</b>
            </span>
          </div>
        </div>

        {/* Subplot 1: Respiration Force (N) Oscilloscope */}
        <div className="mb-5">
          <div className="text-xs font-semibold text-slate-700 mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-blue-600"></span>
              Live Respiration Chest Force (N)
            </span>
            <span className="text-blue-700 font-extrabold text-sm font-mono">{latestForce.toFixed(2)} N</span>
          </div>
          <div className="relative w-full h-44 rounded-xl border border-slate-200 overflow-hidden bg-white shadow-2xs">
            <canvas ref={forceCanvasRef} className="w-full h-full block" />
          </div>
        </div>

        {/* Subplot 2: Breathing Rate (BPM) Trajectory */}
        <div className="mb-5">
          <div className="text-xs font-semibold text-slate-700 mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-emerald-600" />
              Breathing Rate (BPM) vs. Personal Baseline
            </span>
            <span className="text-emerald-700 font-extrabold text-sm font-mono">
              {latestBpm ? `${latestBpm.toFixed(1)} BPM` : '—'}
            </span>
          </div>
          <div className="relative w-full h-36 rounded-xl border border-slate-200 overflow-hidden bg-white shadow-2xs">
            <canvas ref={bpmCanvasRef} className="w-full h-full block" />
          </div>
        </div>
      </div>

      {/* Primary Telemetry Stat Cards */}
      <div className="grid grid-cols-3 gap-3.5 pt-4 border-t border-slate-200">
        <div className="bg-slate-50 border border-slate-200/80 p-3.5 rounded-xl">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Belt Force</div>
          <div className="text-3xl font-extrabold font-mono text-blue-700 mt-1">
            {latestForce.toFixed(2)} <span className="text-sm font-normal text-slate-400">N</span>
          </div>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 p-3.5 rounded-xl">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5 text-emerald-600" /> Breathing Rate
          </div>
          <div className="text-3xl font-extrabold font-mono text-emerald-700 mt-1">
            {latestBpm !== null ? (
              <>
                {latestBpm.toFixed(1)} <span className="text-sm font-normal text-slate-400">BPM</span>
              </>
            ) : (
              <span className="text-slate-400 text-sm font-medium">Measuring...</span>
            )}
          </div>
        </div>

        <div className="bg-slate-50 border border-slate-200/80 p-3.5 rounded-xl">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Baseline Calm</div>
          <div className="text-3xl font-extrabold font-mono text-slate-800 mt-1">
            {baselineBpm !== null ? (
              <>
                {baselineBpm.toFixed(1)} <span className="text-sm font-normal text-slate-400">BPM</span>
              </>
            ) : (
              <span className="text-amber-600 text-sm font-semibold">Pending Calib</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
