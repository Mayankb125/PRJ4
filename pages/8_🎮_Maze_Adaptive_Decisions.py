"""Page 8: ONE live dashboard - respiration/stress detection + Unity logs + how the maze adapts.

Read-only. Nothing here changes Unity or the ML. Sources:
  * Python live manager (same singleton as page 7): belt waveform, breathing rate, window progress, Unity link
  * Python predictions_*.csv : one row per ML window (2 minutes in the headset study) (High / Low, confidence, bpm)
  * Unity  trials.csv        : one row per maze trial incl. the adaptive decision
  * Unity  events.csv        : every event (baseline, trial start/end, adapt, hazards, ...)
Unity folder: Users/<you>/AppData/LocalLow/DefaultCompany/MazeStudy/MazeStudy/<participant>_<stamp>/
"""
from pathlib import Path
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).resolve().parent.parent
RSP_MODEL_DIR = ROOT_DIR / "RSP_Only_Model"
SESSIONS_DIR = RSP_MODEL_DIR / "live_sessions"
if str(RSP_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(RSP_MODEL_DIR))

from live_rsp_ml_manager import LiveRSPMLManager, get_unity_link_status  # same singleton as page 7

st.set_page_config(page_title="VR Adaptive Dashboard", page_icon="🎮", layout="wide")

DEFAULT_UNITY_ROOT = Path.home() / "AppData" / "LocalLow" / "DefaultCompany" / "MazeStudy" / "MazeStudy"
# where tools/quest_usb/quest_usb_bridge.ps1 copies the standalone Quest's logs (adb pull)
QUEST_PULL_ROOT = Path(r"D:\Vr_Project\MazeStudy\tools\quest_usb\pulled\MazeStudy")

CHANGE_STYLE = {
    "level_up": ("▲ Level up", "#16a34a", "triangle-up"),
    "guidance_down": ("▲ Less help", "#65a30d", "triangle-up"),
    "level_down": ("▼ Level down", "#dc2626", "triangle-down"),
    "guidance_up": ("▼ More help", "#ea580c", "triangle-down"),
    "at_limit": ("■ At limit", "#6b7280", "square"),
    "none": ("● No change", "#9ca3af", "circle"),
}
EVENT_ICON = {
    "session_start": "🟢", "session_end": "🏁", "session_abort": "⛔", "block_start": "📦", "block_end": "📦",
    "baseline_start": "🧘", "baseline_end": "✅", "trial_start": "▶️", "trial_end": "⏹️", "adapt": "⚙️",
    "hazards": "⚠️", "hazard_hit": "💥", "gates": "🚪", "gate_block": "🚧", "rest_start": "☕", "rest_end": "☕",
    "fms_score": "📝", "questionnaire_marker": "📝", "jolt": "💥", "hint_denied": "🚫", "error": "❌",
    "ml_command": "📡", "ml_ack": "📥",   # headset -> Python ("start recording") and Python's reply
}
NOISY_EVENTS = {"move"}   # hundreds per trial - counted, not listed


# ----------------------------------------------------------------------------- data loading
def to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def has_rows(path: Path) -> bool:
    """True if a CSV has at least one data row (not just the header)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            f.readline()
            return bool(f.readline().strip())
    except OSError:
        return False


def list_unity_sessions(root: Path) -> list:
    if not root.is_dir():
        return []
    sessions = [p for p in root.iterdir() if p.is_dir() and (p / "trials.csv").is_file()]
    return sorted(sessions, key=lambda p: max((p / n).stat().st_mtime for n in ("trials.csv", "events.csv") if (p / n).is_file()),
                  reverse=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def load_trials(session: Path) -> pd.DataFrame:
    df = read_csv_safe(session / "trials.csv")
    if df.empty:
        return df
    df["end_ts"] = to_ts(df["end_utc"])
    df["start_ts"] = to_ts(df["start_utc"])
    for c in ("reason", "change", "rule_cell", "adapt_mode", "physio_state", "physio_status", "perf_class", "outcome"):
        df[c] = df[c].fillna("").astype(str)
    return df


def load_events(session: Path) -> pd.DataFrame:
    df = read_csv_safe(session / "events.csv")
    if df.empty:
        return df
    df["ts"] = to_ts(df["utc"])
    df["type"] = df["type"].fillna("").astype(str)
    df["detail"] = df["detail"].fillna("").astype(str)
    return df


def list_prediction_files() -> list:
    if not SESSIONS_DIR.is_dir():
        return []
    return sorted((p for p in SESSIONS_DIR.glob("predictions_*.csv") if has_rows(p)),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def load_windows(path) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    df = read_csv_safe(path)
    if df.empty:
        return df
    df["ts"] = to_ts(df["timestamp_utc"])
    # Unity is told "Baseline / Unknown" for the baseline window, never a real High/Low
    df["ml_state"] = df.apply(
        lambda r: "Baseline" if str(r.get("is_baseline_window", "")).lower() == "true" else str(r["vr_state"]), axis=1)
    return df



ACTION_WORDS = {   # what a rule means, in plain words (colour = direction of the change)
    "StepUp": ("Step UP now<br><small>less help, then harder level</small>", "#16a34a"),
    "StepUpIfRepeated": ("Step UP after 2 in a row<br><small>less help, then harder level</small>", "#65a30d"),
    "StepDown": ("Step DOWN<br><small>more help, then easier level</small>", "#dc2626"),
    "Hold": ("HOLD<br><small>no change</small>", "#6b7280"),
}
RULE_KEYS = {"HG": "HIGH_GOOD", "HP": "HIGH_POOR", "LG": "LOW_GOOD", "LP": "LOW_POOR", "PG": "PERF_GOOD", "PP": "PERF_POOR"}
ORIGINAL_RULES = {"HIGH_GOOD": "StepUp", "HIGH_POOR": "StepDown", "LOW_GOOD": "StepUpIfRepeated", "LOW_POOR": "StepDown",
                  "PERF_GOOD": "StepUpIfRepeated", "PERF_POOR": "StepDown"}


def rules_in_use(session):
    """The rule table the headset REALLY used: parsed from the 'adaptive_block*' row Unity writes to session.csv."""
    rules, src = dict(ORIGINAL_RULES), "original plan (no settings row found in session.csv)"
    if session is not None:
        df = read_csv_safe(session / "session.csv")
        if df is not None and not df.empty and "key" in df.columns:
            rows_ = df[df["key"].astype(str).str.startswith("adaptive_block")]
            if not rows_.empty:
                m = re.search(r"rules=([^;]+)", str(rows_.iloc[-1]["value"]))
                if m:
                    for part in m.group(1).split(","):
                        k, _, v = part.partition(":")
                        if k in RULE_KEYS:
                            rules[RULE_KEYS[k]] = v
                    src = "logged by the headset (" + str(rows_.iloc[-1]["key"]) + ")"
    return rules, src


def render_matrix(rules, src, counts, active):
    def cell(name):
        words, color = ACTION_WORDS.get(rules.get(name, "?"), (str(rules.get(name, "?")), "#6b7280"))
        border = "4px solid #111827" if name == active else "1px solid #9ca3af"
        n = counts.get(name, 0)
        return (f"<td style='border:{border};padding:10px;text-align:center;background:{color}26;width:38%'>"
                f"<b style='color:{color}'>{words}</b><br><span style='font-size:12px'>{name}: {n} trial(s)</span></td>")
    html = ("<table style='width:100%;border-collapse:collapse'>"
            "<tr><th style='width:24%'></th><th>Performance GOOD</th><th>Performance POOR</th></tr>"
            f"<tr><th>Stress HIGH</th>{cell('HIGH_GOOD')}{cell('HIGH_POOR')}</tr>"
            f"<tr><th>Stress LOW</th>{cell('LOW_GOOD')}{cell('LOW_POOR')}</tr>"
            f"<tr><th>ML missing / low confidence</th>{cell('PERF_GOOD')}{cell('PERF_POOR')}</tr></table>")
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Rules {src}. Black border = the cell used by the latest decision. 'Step UP' = take away guidance first, then raise the level; "
               "'Step DOWN' = add guidance first (up to G3), then lower the level.")


# ----------------------------------------------------------------------------- explanations / matching
def explain(row: pd.Series) -> str:
    """One plain-English sentence for a trial's adaptive decision."""
    if int(row.get("practice", 0)) == 1:
        return "Practice trial - no adaptation."
    cond = str(row.get("adapt_condition", ""))
    if cond == "FIXED":
        return "FIXED condition - the maze never changes."
    perf = "performed well" if row["perf_class"] == "Good" else "struggled"
    if row["outcome"] and row["outcome"] != "Success":
        perf += " (trial " + row["outcome"].lower() + ")"
    if row["adapt_mode"] == "combined":
        ml = f"ML says {row['physio_state'].upper()} ({float(row['physio_confidence']):.0%} sure)"
    elif cond == "PERF_ONLY":
        ml = "ML ignored (performance-only condition)"
    else:
        ml = f"ML not used ({row['physio_status']})"
    pl, pg, nl, ng = int(row["prev_level"]), int(row["prev_guidance"]), int(row["next_level"]), int(row["next_guidance"])
    change, reason = row["change"], row["reason"]
    if change == "level_up":
        result = f"-> next maze is HARDER (level {pl} -> {nl})"
    elif change == "guidance_down":
        result = f"-> help REMOVED (guidance G{pg} -> G{ng})"
    elif change == "level_down":
        result = f"-> next maze is EASIER (level {pl} -> {nl}, guidance reset to G{ng})"
    elif change == "guidance_up":
        result = f"-> help ADDED (guidance G{pg} -> G{ng})"
    elif change == "at_limit":
        result = "-> wanted to change but already at the limit"
    elif "cooldown" in reason:
        result = "-> held (cooldown after a recent change)"
    elif "waiting_for_confirmation" in reason:
        result = "-> held (waiting for the same result again to confirm)"
    else:
        result = "-> held (rule says no change)"
    return f"{row['perf_class']}: participant {perf}; {ml} {result}"


def match_window(trials: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Which ML window was the newest when each trial ended, and does Unity agree with it?"""
    t = trials.copy()
    t["ml_window"] = pd.NA
    t["ml_says"] = ""
    t["agree"] = ""
    if windows.empty or t.empty:
        return t
    w = windows.sort_values("ts")
    for i, tr in t.iterrows():
        if pd.isna(tr["end_ts"]):
            continue
        prior = w[w["ts"] <= tr["end_ts"] + pd.Timedelta(seconds=2)]
        if prior.empty:
            t.at[i, "ml_says"] = "no window yet"
            continue
        last = prior.iloc[-1]
        t.at[i, "ml_window"] = int(last["window_index"])
        t.at[i, "ml_says"] = last["ml_state"]
        unity = tr["physio_state"] if tr["physio_status"] == "Valid" else "Unknown"
        expected = "Unknown" if last["ml_state"] == "Baseline" else last["ml_state"]
        if tr["physio_status"] in ("Disconnected", "Stale"):
            t.at[i, "agree"] = "Unity: " + tr["physio_status"].lower()
        else:
            t.at[i, "agree"] = "match" if unity.lower() == expected.lower() else "DIFFERENT"
    return t


def unity_phase(events: pd.DataFrame):
    """(phase text, age of newest event in s or None, kind) from Unity's events.csv."""
    if events.empty:
        return "No Unity session data yet - press Play in Unity", None, "none"
    age = None
    last_ts = events["ts"].dropna()
    if not last_ts.empty:
        age = (pd.Timestamp.now(tz="UTC") - last_ts.iloc[-1]).total_seconds()
    marks = events[events["type"].isin(["session_end", "session_abort", "baseline_start", "baseline_end", "trial_start",
                                        "trial_end", "rest_start", "rest_end", "block_start", "block_end", "session_start"])]
    if marks.empty:
        return "Session running", age, "run"
    m = marks.iloc[-1]
    t, d = m["type"], m["detail"]
    if t in ("session_end", "session_abort"):
        return "Session ended" + (" (aborted)" if t == "session_abort" else ""), age, "end"
    if t == "baseline_start":
        planned = ""
        for part in d.split(";"):
            if part.startswith("planned_s="):
                planned = part.split("=", 1)[1]
        return f"Resting baseline in progress ({planned} s planned) - participant should sit still", age, "baseline"
    if t == "trial_start":
        return f"Maze trial {int(m['trial'])} in progress ({d.split(';practice')[0]})", age, "trial"
    if t == "rest_start":
        return "Rest break", age, "run"
    return "Between trials / waiting for the next maze", age, "run"


# ----------------------------------------------------------------------------- sidebar
st.title("🎮 VR Adaptive Dashboard - stress detection × Unity × level changes")
st.caption("Everything in one place. Read-only; refreshes every 2 s. Start the belt on page 7, press Play in Unity.")

with st.sidebar:
    st.header("Sources")
    # pick the source whose newest session was STARTED most recently (folder names end in _YYYYMMDD_HHMMSS)
    def _newest_stamp(root: Path) -> str:
        stamps = [re.search(r"_(\d{8}_\d{6})$", p.name) for p in list_unity_sessions(root)]
        return max((m.group(1) for m in stamps if m), default="")
    _auto = 1 if _newest_stamp(QUEST_PULL_ROOT) > _newest_stamp(DEFAULT_UNITY_ROOT) else 0
    unity_src = st.radio("Unity is running on", ["PC (editor / Quest Link)", "Quest standalone (USB pull)"], index=_auto,
                         help="Auto-selected from the newest session. Quest standalone: start tools/quest_usb/start_quest_usb_bridge.bat, "
                              "it copies the headset's logs to the PC.")
    default_root = DEFAULT_UNITY_ROOT if unity_src.startswith("PC") else QUEST_PULL_ROOT
    unity_root = Path(st.text_input("Unity log folder", str(default_root), key="root_" + unity_src[:5]))
    sessions = list_unity_sessions(unity_root)
    follow = st.checkbox("Follow newest Unity session", value=True)
    chosen = None
    if sessions:
        idx = 0 if follow else st.selectbox("Unity session", range(len(sessions)), format_func=lambda i: sessions[i].name)
        chosen = sessions[idx]
        if follow:
            st.caption(f"Session: `{chosen.name}`")
    else:
        st.warning("No Unity session folders found here.")

    pred_files = list_prediction_files()
    pred_choice = st.selectbox("ML predictions file", ["(current live session / newest)"] + [p.name for p in pred_files])
    show_practice = st.checkbox("Show practice trials", value=False)
    refresh = st.checkbox("Auto-refresh (2 s)", value=True)


# ----------------------------------------------------------------------------- the dashboard
@st.fragment(run_every=2 if refresh else None)
def live_view():
    manager = LiveRSPMLManager.get_instance()
    snap = manager.get_snapshot()
    link = get_unity_link_status()

    pred_path = None
    if pred_choice != "(current live session / newest)":
        pred_path = SESSIONS_DIR / pred_choice
    else:
        live_path = Path(snap["predictions_file"]) if snap.get("predictions_file") else None
        if live_path is not None and live_path.is_file() and has_rows(live_path):
            pred_path = live_path
        elif pred_files:
            pred_path = pred_files[0]
    windows = load_windows(pred_path)
    # window length actually used (from the file), else the live manager's setting
    wl = int(round(float(windows.iloc[-1]["window_end_seconds"]) - float(windows.iloc[-1]["window_start_seconds"]))) if not windows.empty else int(snap["window_duration_s"])

    trials_all = load_trials(chosen) if chosen else pd.DataFrame()
    events = load_events(chosen) if chosen else pd.DataFrame()
    trials = trials_all if show_practice or trials_all.empty else trials_all[trials_all["practice"] == 0]
    trials = match_window(trials, windows)
    if not trials.empty:
        trials["explanation"] = trials.apply(explain, axis=1)

    phase_text, ev_age, phase_kind = unity_phase(events)

    # ---------- banner: what is happening right now
    if phase_kind == "baseline":
        st.warning("🧘 " + phase_text)
    elif phase_kind == "trial":
        st.info("▶️ " + phase_text)
    elif phase_kind == "end":
        st.success("🏁 " + phase_text)
    else:
        st.info("⏳ " + phase_text)

    # ---------- 1. status strip
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    belt = "streaming" if snap["is_running"] else ("connecting..." if snap["is_connecting"] else "not started")
    c1.metric("Belt (page 7)", belt)
    c2.metric("Unity ↔ ML link", "connected" if link["clients"] > 0 else ("waiting for Unity" if link["status"] == "listening" else link["status"]),
              help=f"TCP {link['address']}")
    if ev_age is None:
        c3.metric("Unity activity", "no data")
    else:
        c3.metric("Unity activity", "live" if ev_age < 15 else ("idle" if ev_age < 90 else "silent"), f"last event {ev_age:.0f} s ago",
                  delta_color="off", help="Newest line in Unity's events.csv. If the headset app is running, this keeps updating.")
    if snap["is_running"]:
        c4.metric("ML window", f"{snap['current_window_elapsed']:.0f} / {snap['window_duration_s']:.0f} s",
                  help="A new High/Low answer is sent to Unity each time this fills up")
    else:
        c4.metric("ML window", "-")
    if not windows.empty:
        lw = windows.iloc[-1]
        c5.metric("Latest ML answer", lw["ml_state"], f"{float(lw['confidence']):.0%} conf", delta_color="off")
    else:
        c5.metric("Latest ML answer", "none yet")
    if not trials_all.empty:
        last = trials_all.iloc[-1]
        c6.metric("Maze next", f"L{int(last['next_level'])} · G{int(last['next_guidance'])}", help="Level / guidance chosen for the next trial")
    else:
        c6.metric("Maze next", "-")

    # ---------- 2. live stress detection
    st.subheader("1 · Stress detection (respiration → ML)")
    left, right = st.columns([2.1, 1])
    with left:
        data = snap.get("data") or []
        if snap["is_running"] and len(data) > 5:
            d = pd.DataFrame(data)
            d["t"] = d["elapsed"] - d["elapsed"].iloc[-1]
            d = d[d["t"] >= -90]
            f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                               subplot_titles=("Respiration belt force (N)", "Breathing rate (bpm) vs. personal baseline"))
            f2.add_trace(go.Scatter(x=d["t"], y=d["force"], mode="lines", line=dict(color="#0ea5e9", width=1.5), showlegend=False), row=1, col=1)
            b = d.dropna(subset=["bpm"])
            f2.add_trace(go.Scatter(x=b["t"], y=b["bpm"], mode="lines+markers", line=dict(color="#7c3aed"), showlegend=False), row=2, col=1)
            if snap.get("baseline_bpm") is not None:
                f2.add_hline(y=snap["baseline_bpm"], line_dash="dot", line_color="#16a34a", row=2, col=1,
                             annotation_text=f"baseline {snap['baseline_bpm']:.1f}", annotation_position="top left")
                f2.add_hline(y=snap["baseline_bpm"] + 2.5, line_dash="dash", line_color="#dc2626", row=2, col=1,
                             annotation_text="+2.5 bpm: rate rule starts to count", annotation_position="bottom left")
            f2.update_xaxes(title_text="seconds ago", row=2, col=1)
            f2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(f2, use_container_width=True)
        else:
            st.info("Belt is not streaming. Open page 7 and click **Start** to see the live respiration waveform here.")
    with right:
        if not windows.empty:
            lw = windows.iloc[-1]
            prob = float(lw["stress_probability"])
            is_base = lw["ml_state"] == "Baseline"
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": " %"},
                title={"text": f"Stress probability (last {wl} s window)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#dc2626" if prob >= 0.5 else "#2563eb"},
                       "steps": [{"range": [0, 50], "color": "#dbeafe"}, {"range": [50, 100], "color": "#fee2e2"}],
                       "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            gauge.update_layout(height=240, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(gauge, use_container_width=True)
            if is_base:
                st.caption("🧘 Window #{} set the personal baseline (Unity gets *Baseline*, not High/Low).".format(int(lw["window_index"])))
            else:
                st.markdown(f"**Window {int(lw['window_index'])}: {lw['ml_state']}** · {float(lw['calculated_bpm']):.1f} bpm "
                            f"(Δ {float(lw['delta_bpm']):+.1f} vs baseline)")
                parts = []
                if "rf_stress_probability" in lw:
                    parts.append(f"RF model {float(lw['rf_stress_probability']):.0%}")
                if "rate_stress_probability" in lw:
                    parts.append(f"rate rule {float(lw['rate_stress_probability']):.0%}")
                if parts:
                    st.caption(" · ".join(parts) + f" → decided by **{lw.get('decision_source', '?')}**")
        else:
            st.info(f"No ML window finished yet. The first one (the first {wl} s) becomes the baseline.")

    # ---------- 3. latest decision
    st.subheader("2 · How the maze reacted")
    if trials.empty:
        st.info("No finished trials yet. Finish a maze in Unity - it shows up here within 2 seconds.")
    else:
        lt = trials.iloc[-1]
        st.markdown(f"**Latest decision - trial {int(lt['global_trial'])}** (played level {int(lt['level'])}, guidance G{int(lt['guidance'])}, "
                    f"condition `{lt.get('adapt_condition', '')}`)")
        (st.success if lt["change"] != "none" else st.info)(lt["explanation"])

        # the stress x performance matrix that is actually in force, with the cell of the latest decision highlighted
        adaptive_rows = trials[(trials["practice"] == 0)] if "practice" in trials.columns else trials
        rules_now, rules_src = rules_in_use(chosen)
        st.markdown("**Decision matrix in use (stress × performance)**")
        render_matrix(rules_now, rules_src, adaptive_rows["rule_cell"].value_counts().to_dict(),
                      lt["rule_cell"] if lt["rule_cell"] else "")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.34, 0.33, 0.33],
                            subplot_titles=(f"ML: stress probability per {wl} s window (≥ 50% = High)",
                                            "Maze level of each trial (▲ harder / ▼ easier)",
                                            "Guidance level (G0 = no help … G5 = most help)"))
        if not windows.empty:
            colors = ["#2563eb" if s == "Low" else "#dc2626" if s == "High" else "#9ca3af" for s in windows["ml_state"]]
            fig.add_trace(go.Bar(x=windows["ts"], y=windows["stress_probability"], marker_color=colors, opacity=0.85, name="ML window",
                                 showlegend=False, text=windows["ml_state"], textposition="inside",
                                 hovertext=[f"window {w}: {s}, conf {c:.0%}, {b:.1f} bpm" for w, s, c, b in
                                            zip(windows["window_index"], windows["ml_state"], windows["confidence"], windows["calculated_bpm"])],
                                 hoverinfo="text"), row=1, col=1)
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        xs = trials["end_ts"]
        fig.add_trace(go.Scatter(x=xs, y=trials["level"], mode="lines", line=dict(shape="hv", color="#111827"), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=trials["guidance"], mode="lines", line=dict(shape="hv", color="#7c3aed"), showlegend=False), row=3, col=1)
        for change, (label, color, symbol) in CHANGE_STYLE.items():
            sub = trials[trials["change"] == change]
            if sub.empty:
                continue
            for row_i, col_name in ((2, "level"), (3, "guidance")):
                fig.add_trace(go.Scatter(
                    x=sub["end_ts"], y=sub[col_name], mode="markers", name=label, legendgroup=label, showlegend=(row_i == 2),
                    marker=dict(size=13, color=color, symbol=symbol, line=dict(width=1, color="white")),
                    hovertext=[f"trial {g}: {e}" for g, e in zip(sub["global_trial"], sub["explanation"])], hoverinfo="text"),
                    row=row_i, col=1)
        fig.update_yaxes(dtick=1, row=2, col=1)
        fig.update_yaxes(dtick=1, range=[-0.5, 5.5], row=3, col=1)
        fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True)

        table = pd.DataFrame({
            "trial": trials["global_trial"],
            "ended (UTC)": trials["end_ts"].dt.strftime("%H:%M:%S"),
            "level": trials["level"].astype(str) + " → " + trials["next_level"].astype(str),
            "guidance": "G" + trials["guidance"].astype(str) + " → G" + trials["next_guidance"].astype(str),
            "performance": trials["perf_class"],
            "time vs ref": trials["speed_ratio"].round(2),
            "ML window": trials["ml_window"],
            "ML says (window)": trials["ml_says"],
            "Unity used": trials["physio_status"] + " / " + trials["physio_state"] + " (" + trials["physio_confidence"].round(2).astype(str) + ")",
            "check": trials["agree"],
            "rule": trials["rule_cell"],
            "result": trials["change"].map(lambda c: CHANGE_STYLE.get(c, (c,))[0]),
            "why": trials["explanation"],
        })

        def color_result(v):
            for _, (label, color, _s) in CHANGE_STYLE.items():
                if v == label:
                    return f"color: {color}; font-weight: 700"
            return ""

        st.dataframe(table.style.map(color_result, subset=["result"])
                     .map(lambda v: "color: #dc2626; font-weight: 700" if v == "DIFFERENT" else "", subset=["check"]),
                     hide_index=True, use_container_width=True, height=min(60 + 36 * len(table), 480))
        st.caption("**check** compares what Unity used for the trial with the ML window that was newest when it ended; `DIFFERENT` = look for a "
                   "delayed window or a link drop. Unity uses the ML only when status is Valid and confidence ≥ 0.6, else performance-only rules.")

    # ---------- 4. Unity live log feed
    st.subheader("3 · Unity logs (live)")
    f1, f2c = st.columns([2, 1])
    with f1:
        if events.empty:
            st.info("No Unity events yet.")
        else:
            shown = events[~events["type"].isin(NOISY_EVENTS)].tail(30).iloc[::-1]
            feed = pd.DataFrame({
                "UTC": shown["ts"].dt.strftime("%H:%M:%S"),
                "session s": shown["session_s"].round(1),
                "event": shown["type"].map(lambda t: EVENT_ICON.get(t, "•") + " " + t),
                "trial": shown["trial"],
                "detail": shown["detail"].str.slice(0, 110),
            })
            st.dataframe(feed, hide_index=True, use_container_width=True, height=380)
            moves = int((events["type"] == "move").sum())
            st.caption(f"Newest first · {len(events)} events in this session ({moves} `move` events hidden) · file: `{chosen.name}/events.csv`")
    with f2c:
        if not trials.empty:
            st.markdown("**Decisions by rule**")
            st.dataframe(trials.groupby(["rule_cell", "change"]).size().rename("trials").reset_index(), hide_index=True, use_container_width=True)
            st.markdown("**ML used or not**")
            st.dataframe(trials.groupby("adapt_mode").size().rename("trials").reset_index().rename(columns={"adapt_mode": "mode"}),
                         hide_index=True, use_container_width=True)
            st.caption("`combined` = ML + performance · `performance_only` = ML missing, low confidence or switched off.")


live_view()
