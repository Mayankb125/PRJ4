import React from 'react';
import { UnityLatest } from '../types';
import {
  Gamepad2,
  ArrowUpCircle,
  ArrowDownCircle,
  ShieldAlert,
  Compass,
  Trophy,
  History,
  Grid2X2,
} from 'lucide-react';

interface VRAdaptivePanelProps {
  unityData: UnityLatest;
  unityLink: {
    status: string;
    clients: number;
    last_error: string | null;
  };
}

export const VRAdaptivePanel: React.FC<VRAdaptivePanelProps> = ({
  unityData,
  unityLink,
}) => {
  const latestTrial = unityData?.latest_trial;
  const isConnected = unityLink?.clients > 0;
  const rules = unityData?.rules || {};
  const cellCounts = unityData?.cell_counts || {};
  const activeCell = unityData?.active_cell || latestTrial?.rule_cell;
  const recentEvents = unityData?.recent_events || [];

  const isLevelUp = latestTrial?.change === 'level_up';
  const isLevelDown = latestTrial?.change === 'level_down';
  const isGuidanceUp = latestTrial?.change === 'guidance_up';
  const isGuidanceDown = latestTrial?.change === 'guidance_down';

  const currentLevel = latestTrial?.next_level ?? 1;
  const currentGuidance = latestTrial?.next_guidance ?? 0;

  const normalizeKey = (key: string) => {
    const k = key.toUpperCase().replace(/\s+/g, '_');
    if (k.includes('HIGH') && (k.includes('GOOD') || k.includes('HIGH_PERF'))) return 'HIGH_GOOD';
    if (k.includes('HIGH') && (k.includes('POOR') || k.includes('LOW_PERF'))) return 'HIGH_POOR';
    if (k.includes('LOW') && (k.includes('GOOD') || k.includes('HIGH_PERF'))) return 'LOW_GOOD';
    if (k.includes('LOW') && (k.includes('POOR') || k.includes('LOW_PERF'))) return 'LOW_POOR';
    if (k.includes('PERF') && (k.includes('GOOD') || k.includes('HIGH_PERF'))) return 'PERF_GOOD';
    if (k.includes('PERF') && (k.includes('POOR') || k.includes('LOW_PERF'))) return 'PERF_POOR';
    return k;
  };

  const getActionInfo = (ruleVal: string) => {
    switch (ruleVal) {
      case 'StepUp':
        return {
          text: 'Step UP now',
          sub: 'Less help, then harder maze level',
          badge: 'bg-emerald-100 text-emerald-800 border-emerald-300',
          bg: 'hover:bg-emerald-50/50',
        };
      case 'StepUpIfRepeated':
        return {
          text: 'Step UP (2 in row)',
          sub: 'Require 2 consecutive good trials',
          badge: 'bg-lime-100 text-lime-800 border-lime-300',
          bg: 'hover:bg-lime-50/50',
        };
      case 'StepDown':
        return {
          text: 'Step DOWN',
          sub: 'More help, then easier maze level',
          badge: 'bg-rose-100 text-rose-800 border-rose-300',
          bg: 'hover:bg-rose-50/50',
        };
      case 'Hold':
      default:
        return {
          text: 'HOLD Level',
          sub: 'Maintain current maze difficulty',
          badge: 'bg-slate-100 text-slate-700 border-slate-300',
          bg: 'hover:bg-slate-50',
        };
    }
  };

  const renderCell = (cellKey: string, aliasKeys: string[] = []) => {
    const norm = normalizeKey(cellKey);
    const actionVal = rules[norm] || rules[cellKey] || 'Hold';
    const info = getActionInfo(actionVal);

    // Sum trial counts for this cell and its aliases
    let count = cellCounts[norm] || cellCounts[cellKey] || 0;
    for (const a of aliasKeys) {
      count += (cellCounts[a] || 0);
    }

    const currentActiveNorm = activeCell ? normalizeKey(activeCell) : '';
    const isActive = currentActiveNorm === norm || activeCell === cellKey || aliasKeys.includes(activeCell || '');

    return (
      <td
        className={`p-4 sm:p-5 text-center border transition-all ${
          isActive
            ? 'border-3 border-blue-600 bg-blue-50/90 shadow-md ring-4 ring-blue-500/20'
            : `border-slate-200/90 bg-white ${info.bg}`
        }`}
      >
        <div className="flex flex-col items-center justify-center gap-1.5">
          <span className={`inline-block px-3 py-1 rounded-full text-xs sm:text-sm font-extrabold border shadow-2xs ${info.badge}`}>
            {info.text}
          </span>
          <span className="text-[11px] text-slate-500 font-medium">
            {info.sub}
          </span>
          <span className="text-xs font-semibold text-slate-600 bg-slate-100/80 px-2.5 py-0.5 rounded-md border border-slate-200 mt-1">
            <b className="text-slate-900 font-mono text-sm">{count}</b> trial{count === 1 ? '' : 's'} recorded
          </span>
        </div>
      </td>
    );
  };

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-sm">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between pb-4 border-b border-slate-200 mb-5 gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-sky-50 border border-sky-200 flex items-center justify-center shadow-xs">
            <Gamepad2 className="w-5 h-5 text-sky-600" />
          </div>
          <div>
            <h2 className="text-lg font-bold font-display tracking-tight text-slate-900">
              VR Headset Adaptive Bridge (Quest TCP :5005)
            </h2>
            <div className="text-xs text-slate-500 font-medium">
              Meta Quest MazeStudy Closed-Loop Difficulty & Guidance Control
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 text-xs">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              isConnected ? 'bg-emerald-500 animate-ping' : 'bg-slate-400'
            }`}
          />
          <span className="bg-slate-100 px-3 py-1.5 rounded-lg text-slate-700 font-semibold border border-slate-200">
            {isConnected ? `Quest Connected (${unityLink.clients} Client)` : 'TCP Port 5005 Listening'}
          </span>
          {unityData.session_name && (
            <span className="bg-blue-50 text-blue-700 px-3 py-1.5 rounded-lg font-mono text-xs border border-blue-200">
              Session: {unityData.session_name}
            </span>
          )}
        </div>
      </div>

      {/* Dynamic Adaptation Alerts (Prominent Visual Banners) */}
      {isLevelUp && (
        <div className="mb-5 p-4 bg-emerald-50 border-2 border-emerald-400 rounded-xl flex items-center justify-between text-emerald-950 shadow-sm animate-fadeIn">
          <div className="flex items-center gap-3">
            <ArrowUpCircle className="w-7 h-7 text-emerald-600 shrink-0" />
            <div>
              <div className="text-sm font-extrabold uppercase tracking-wide text-emerald-800">
                Difficulty Level Up Triggered
              </div>
              <div className="text-xs text-emerald-700 font-medium mt-0.5">
                Participant completed the maze quickly with low physiological stress. Advancing to{' '}
                <b>Level {latestTrial.next_level}</b>.
              </div>
            </div>
          </div>
          <span className="text-xs font-mono font-bold bg-white px-3 py-1.5 rounded-lg text-emerald-800 border border-emerald-300">
            Level {latestTrial.prev_level} → {latestTrial.next_level}
          </span>
        </div>
      )}

      {isLevelDown && (
        <div className="mb-5 p-4 bg-rose-50 border-2 border-rose-400 rounded-xl flex items-center justify-between text-rose-950 shadow-sm animate-fadeIn">
          <div className="flex items-center gap-3">
            <ArrowDownCircle className="w-7 h-7 text-rose-600 shrink-0" />
            <div>
              <div className="text-sm font-extrabold uppercase tracking-wide text-rose-800">
                Difficulty Level Down Triggered
              </div>
              <div className="text-xs text-rose-700 font-medium mt-0.5">
                Participant experienced high stress and task difficulty. Decreasing to{' '}
                <b>Level {latestTrial.next_level}</b> to reduce cognitive load.
              </div>
            </div>
          </div>
          <span className="text-xs font-mono font-bold bg-white px-3 py-1.5 rounded-lg text-rose-800 border border-rose-300">
            Level {latestTrial.prev_level} → {latestTrial.next_level}
          </span>
        </div>
      )}

      {isGuidanceUp && (
        <div className="mb-5 p-4 bg-amber-50 border border-amber-300 rounded-xl flex items-center justify-between text-amber-950 shadow-sm animate-fadeIn">
          <div className="flex items-center gap-3">
            <Compass className="w-6 h-6 text-amber-600 shrink-0" />
            <div>
              <div className="text-sm font-bold text-amber-800">Visual Waypoint Guidance Increased</div>
              <div className="text-xs text-amber-700">
                Guidance cues elevated to <b>G{latestTrial.next_guidance}</b> to assist maze navigation.
              </div>
            </div>
          </div>
          <span className="text-xs font-mono font-bold bg-white px-3 py-1.5 rounded-lg text-amber-800 border border-amber-200">
            G{latestTrial.prev_guidance} → G{latestTrial.next_guidance}
          </span>
        </div>
      )}

      {isGuidanceDown && (
        <div className="mb-5 p-4 bg-blue-50 border border-blue-300 rounded-xl flex items-center justify-between text-blue-950 shadow-sm animate-fadeIn">
          <div className="flex items-center gap-3">
            <Compass className="w-6 h-6 text-blue-600 shrink-0" />
            <div>
              <div className="text-sm font-bold text-blue-800">Visual Waypoint Guidance Reduced</div>
              <div className="text-xs text-blue-700">
                Participant navigating well. Waypoints reduced to <b>G{latestTrial.next_guidance}</b> for greater challenge.
              </div>
            </div>
          </div>
          <span className="text-xs font-mono font-bold bg-white px-3 py-1.5 rounded-lg text-blue-800 border border-blue-200">
            G{latestTrial.prev_guidance} → G{latestTrial.next_guidance}
          </span>
        </div>
      )}

      {unityData.hold_no_physio && (
        <div className="mb-5 p-3.5 bg-amber-50 border border-amber-300 rounded-xl flex items-center gap-3 text-xs sm:text-sm text-amber-900 shadow-xs">
          <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0" />
          <span>
            <b>Sensor Calibration Hold:</b> Belt baseline is still calibrating. VR maze adaptation will hold current difficulty until valid respiration packets arrive.
          </span>
        </div>
      )}

      {/* Main Grid: VR State (4 cols) + Spacious Full-Width Decision Matrix (8 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        {/* Column 1 (4 cols): Current VR State, Level & Adaptation Summary */}
        <div className="lg:col-span-4 space-y-4 flex flex-col justify-between">
          {/* Level & Guidance Hero Card */}
          <div className="bg-slate-50 border border-slate-200/90 rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                Current Maze Difficulty
              </span>
              <span className="text-xs font-bold font-mono px-3 py-1 rounded-full bg-blue-100 text-blue-700">
                Level {currentLevel} of 8
              </span>
            </div>

            {/* Visual 8-Level Stepped Meter */}
            <div className="grid grid-cols-8 gap-1.5 mb-4">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((lvl) => (
                <div
                  key={lvl}
                  className={`h-3 rounded-full transition-all ${
                    lvl <= currentLevel
                      ? 'bg-blue-600 shadow-xs'
                      : 'bg-slate-200'
                  }`}
                  title={`Level ${lvl}`}
                />
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-200/80">
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Guidance Mode</div>
                <div className="text-lg font-bold text-indigo-700 mt-0.5 flex items-center gap-1.5">
                  <Compass className="w-4 h-4" />
                  G{currentGuidance} <span className="text-xs font-normal text-slate-500">/ G5</span>
                </div>
              </div>

              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Last Trial Result</div>
                <div className="text-lg font-bold text-slate-800 mt-0.5 flex items-center gap-1.5">
                  <Trophy className="w-4 h-4 text-amber-500" />
                  {latestTrial ? (
                    <span className="capitalize">{latestTrial.perf_class || 'Success'}</span>
                  ) : (
                    <span className="text-slate-400 text-sm font-medium">In Progress</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Active Decision Logic Summary */}
          <div className="bg-slate-50 border border-slate-200/90 rounded-xl p-5">
            <div className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Latest Closed-Loop Adaptation Rule
            </div>
            {latestTrial ? (
              <div className="text-xs text-slate-700 space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Active Rule Cell:</span>
                  <span className="font-bold font-mono text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    {latestTrial.rule_cell || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Action Applied:</span>
                  <span className="font-bold font-mono capitalize text-slate-900">
                    {latestTrial.change.replace('_', ' ')}
                  </span>
                </div>
                <div className="pt-2 border-t border-slate-200/80 text-xs text-slate-600 italic bg-white p-2.5 rounded-lg border">
                  "{latestTrial.reason || 'Difficulty maintained'}"
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-500 font-medium py-3 text-center">
                Awaiting first trial completion from Meta Quest headset...
              </div>
            )}
          </div>
        </div>

        {/* Column 2 (8 cols): WIDE, SPACIOUS DECISION MATRIX (Stress × Performance) */}
        <div className="lg:col-span-8 bg-slate-50 border border-slate-200/90 rounded-xl p-5">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3 mb-4">
            <div className="flex items-center gap-2">
              <Grid2X2 className="w-4 h-4 text-blue-600" />
              <h3 className="text-sm sm:text-base font-bold text-slate-900">
                Closed-Loop Decision Matrix (Stress × Performance)
              </h3>
            </div>
            <span className="text-xs text-slate-500 font-mono">
              Rules from session.csv
            </span>
          </div>

          {/* Table Container: Full Width, Generous Sizing */}
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse bg-white rounded-xl overflow-hidden shadow-2xs">
              <thead>
                <tr className="bg-slate-100/80 border-b border-slate-200 text-xs text-slate-700">
                  <th className="p-3.5 sm:p-4 font-bold text-slate-500 uppercase tracking-wider w-[28%]">
                    Physiological Stress
                  </th>
                  <th className="p-3.5 sm:p-4 text-center font-bold text-slate-800 uppercase tracking-wider w-[36%]">
                    Performance GOOD <span className="block text-[11px] font-normal text-slate-500">(Fast navigation / Few errors)</span>
                  </th>
                  <th className="p-3.5 sm:p-4 text-center font-bold text-slate-800 uppercase tracking-wider w-[36%]">
                    Performance POOR <span className="block text-[11px] font-normal text-slate-500">(Slow / Obstacle collisions / Timeout)</span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {/* Row 1: High Stress */}
                <tr>
                  <td className="p-3.5 sm:p-4 bg-rose-50/40 border-r border-slate-200">
                    <div className="text-xs sm:text-sm font-bold text-rose-800 flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-rose-600"></span>
                      Stress HIGH
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      Vernier belt & RF classifier
                    </div>
                  </td>
                  {renderCell('HIGH_GOOD', ['high_stress_high_perf'])}
                  {renderCell('HIGH_POOR', ['high_stress_low_perf'])}
                </tr>

                {/* Row 2: Low Stress */}
                <tr>
                  <td className="p-3.5 sm:p-4 bg-emerald-50/40 border-r border-slate-200">
                    <div className="text-xs sm:text-sm font-bold text-emerald-800 flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
                      Stress LOW (Calm)
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      Within resting baseline
                    </div>
                  </td>
                  {renderCell('LOW_GOOD', ['low_stress_high_perf'])}
                  {renderCell('LOW_POOR', ['low_stress_low_perf'])}
                </tr>

                {/* Row 3: Missing / Fallback (Perf Only) */}
                <tr>
                  <td className="p-3.5 sm:p-4 bg-slate-50 border-r border-slate-200">
                    <div className="text-xs sm:text-sm font-bold text-slate-700">
                      Sensor Missing / Calibrating
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      Performance-only fallback
                    </div>
                  </td>
                  {renderCell('PERF_GOOD', ['perf_good'])}
                  {renderCell('PERF_POOR', ['perf_poor'])}
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-3.5 pt-3 border-t border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
            <div className="flex items-center gap-2">
              <span className="inline-block w-3.5 h-3.5 border-2 border-blue-600 bg-blue-50 rounded"></span>
              <span>Luminous blue border indicates the active trial adaptation cell.</span>
            </div>
            <span>'Step UP' / 'Step DOWN' modifies maze difficulty level directly.</span>
          </div>
        </div>
      </div>

      {/* Bottom Sub-Panel: Live Quest VR Event Stream */}
      <div className="bg-slate-50 border border-slate-200/90 rounded-xl p-5">
        <div className="flex items-center justify-between border-b border-slate-200 pb-2.5 mb-3">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase tracking-wider">
            <History className="w-4 h-4 text-blue-600" />
            <span>Live Quest VR Headset Event Stream</span>
          </div>
          <div className="text-xs text-slate-500 font-mono">
            Buffered: <b className="text-slate-800">{recentEvents.length}</b> events · IST Timestamps
          </div>
        </div>

        <div className="space-y-2 max-h-52 overflow-y-auto pr-1">
          {recentEvents.length === 0 ? (
            <div className="text-xs text-slate-500 font-medium py-8 text-center bg-white rounded-lg border border-slate-200">
              No Quest headset events received yet. Start a maze trial in Meta Quest to view real-time events.
            </div>
          ) : (
            recentEvents.map((ev: any, idx: number) => {
              const isDecision = ev.type === 'adaptation_decision' || ev.type === 'adapt';
              const isFinish = ev.type === 'trial_end' || ev.type === 'goal_reached';
              const isStressEv = ev.type === 'reading_applied' || ev.type === 'ml_command';

              return (
                <div
                  key={idx}
                  className="p-2.5 bg-white rounded-lg border border-slate-200 text-xs shadow-2xs flex items-center justify-between gap-3"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                        isDecision
                          ? 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                          : isFinish
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : isStressEv
                          ? 'bg-amber-50 text-amber-700 border border-amber-200'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {ev.type ? ev.type.replace('_', ' ') : 'Event'}
                    </span>
                    <span className="text-slate-800 font-medium truncate max-w-xl">
                      {ev.detail || JSON.stringify(ev.data || '')}
                    </span>
                  </div>

                  <span className="text-xs font-mono text-slate-400 shrink-0">
                    {ev.ist || `${ev.session_s}s`}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
