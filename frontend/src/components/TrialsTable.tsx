import React from 'react';
import { UnityTrial } from '../types';
import { ListFilter, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

interface TrialsTableProps {
  trials: UnityTrial[];
}

export const TrialsTable: React.FC<TrialsTableProps> = ({ trials = [] }) => {
  const safeTrials = trials || [];
  const displayTrials = [...safeTrials].reverse();

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-200 mb-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-50 border border-purple-200 flex items-center justify-center shadow-xs">
            <ListFilter className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h2 className="text-lg font-bold font-display tracking-tight text-slate-900">
              Adaptive Trial Decision Audit Log
            </h2>
            <div className="text-xs text-slate-500 font-medium">
              Closed-loop difficulty adaptations, physiological states, and participant performance
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <span className="text-xs text-slate-600 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200 font-semibold">
            Total Trials: <b className="text-slate-900 font-mono">{safeTrials.length}</b>
          </span>
          <span className="text-xs text-slate-500 bg-slate-50 px-2.5 py-1.5 rounded-lg border border-slate-200 font-medium">
            Newest First (IST)
          </span>
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto max-h-80 overflow-y-auto border border-slate-200 rounded-xl shadow-2xs">
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500 bg-slate-50/90 sticky top-0 text-xs font-semibold uppercase tracking-wider">
              <th className="py-3 px-4">Trial #</th>
              <th className="py-3 px-4">Time (IST)</th>
              <th className="py-3 px-4">Level Transition</th>
              <th className="py-3 px-4">Guidance</th>
              <th className="py-3 px-4">Adaptive Action</th>
              <th className="py-3 px-4">Stress State</th>
              <th className="py-3 px-4">Performance</th>
              <th className="py-3 px-4">Adaptation Reason</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700">
            {displayTrials.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-10 text-center text-slate-500 font-medium">
                  No trial decisions recorded yet for this session. Complete a maze trial in the Quest headset.
                </td>
              </tr>
            ) : (
              displayTrials.map((t) => {
                const isUp = t.change === 'level_up' || t.change === 'guidance_down';
                const isDown = t.change === 'level_down' || t.change === 'guidance_up';
                const isStress = t.physio_state?.toLowerCase() === 'high';

                return (
                  <tr key={t.global_trial} className="hover:bg-slate-50/80 transition">
                    <td className="py-3 px-4 font-mono font-bold text-slate-900">
                      #{t.global_trial}
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-slate-500">
                      {t.end_ist || '--:--:--'}
                    </td>
                    <td className="py-3 px-4 font-semibold text-blue-700">
                      Level {t.prev_level} → <span className="text-slate-900 font-bold">{t.next_level}</span>
                    </td>
                    <td className="py-3 px-4 font-medium text-indigo-700">
                      G{t.prev_guidance} → G{t.next_guidance}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold ${
                          isUp
                            ? 'bg-emerald-50 text-emerald-800 border border-emerald-300'
                            : isDown
                            ? 'bg-rose-50 text-rose-800 border border-rose-300'
                            : 'bg-slate-100 text-slate-700 border border-slate-200'
                        }`}
                      >
                        {isUp && <ArrowUpRight className="w-3 h-3 text-emerald-600" />}
                        {isDown && <ArrowDownRight className="w-3 h-3 text-rose-600" />}
                        {!isUp && !isDown && <Minus className="w-3 h-3 text-slate-400" />}
                        <span className="capitalize">{t.change.replace('_', ' ')}</span>
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center gap-1.5 font-bold text-xs px-2.5 py-1 rounded-full ${
                          isStress
                            ? 'bg-rose-50 text-rose-700 border border-rose-200'
                            : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            isStress ? 'bg-rose-600' : 'bg-emerald-600'
                          }`}
                        />
                        {isStress ? 'High Stress' : 'Calm'}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-900 font-medium">
                      <span className="capitalize">{t.perf_class || 'Success'}</span>
                    </td>
                    <td className="py-3 px-4 text-slate-600 text-xs max-w-sm truncate" title={t.reason}>
                      {t.reason || 'Difficulty maintained'}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
