"""
Java Tutor Study — Interactive Research Dashboard
===================================================
Run with: streamlit run app.py

Loads pre-computed NLP analysis JSON files from figures/nlp/
and the cleaned session JSON from data/

pip install streamlit plotly pandas scikit-learn
"""

import json
import numpy as np
import pandas as pd
import os
os.environ["STREAMLIT_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Java Tutor Study",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0f1117; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-big {
    font-family: 'DM Mono', monospace;
    font-size: 2.4rem;
    font-weight: 500;
    color: #818cf8;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.finding-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #1a1d2e 100%);
    border-left: 3px solid #818cf8;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #e2e8f0;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #ffffff;
    border-bottom: 1px solid rgba(99,102,241,0.3);
    padding-bottom: 8px;
    margin: 32px 0 16px 0;
}
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.pill-high { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
.pill-medium { background: rgba(251,146,60,0.2); color: #fb923c; border: 1px solid rgba(251,146,60,0.4); }
.pill-low { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
COND_COLORS = {
    'character_scaffolded':     '#818cf8',
    'non_character_scaffolded': '#34d399',
    'direct_chat':              '#fb7185',
}
COND_LABELS = {
    'character_scaffolded':     'Char-Scaffolded',
    'non_character_scaffolded': 'Non-Char Scaffolded',
    'direct_chat':              'Direct Chat',
}
TIER_COLORS = {'High': '#4ade80', 'Medium': '#fb923c', 'Low': '#f87171'}
LAYOUT_BASE = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='DM Sans'),
    title_font=dict(color='#ffffff', family='DM Serif Display', size=16),
)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR    = Path("figures/nlp")
SESSION_FILE = Path("data/cleaned_java_session_data_with_queues.json")

@st.cache_data
def load_clusters(sf="all3"):
    f = DATA_DIR / f"nlp_clusters_{sf}.json"
    if not f.exists(): f = DATA_DIR / "nlp_clusters_all3.json"
    return json.load(open(f))

@st.cache_data
def load_response_times(sf="all3"):
    f = DATA_DIR / f"nlp_response_times_{sf}.json"
    if not f.exists(): f = DATA_DIR / "nlp_response_times_all3.json"
    return json.load(open(f))

@st.cache_data
def load_char_counts(sf="all3"):
    f = DATA_DIR / f"nlp_char_counts_{sf}.json"
    if not f.exists(): f = DATA_DIR / "nlp_char_counts_all3.json"
    return json.load(open(f))

@st.cache_data
def load_question_quality(sf="all3"):
    f = DATA_DIR / f"nlp_question_quality_{sf}.json"
    if not f.exists(): f = DATA_DIR / "nlp_question_quality_all3.json"
    return json.load(open(f))

@st.cache_data
def load_session_data():
    if not SESSION_FILE.exists():
        return None
    raw = json.load(open(SESSION_FILE))
    rows = []
    for uid, udata in raw.get("users", {}).items():
        cond = udata.get("condition", "unknown")
        for sname, sdata in udata.get("sessions", {}).items():
            status = sdata.get("status", "")
            if status in ("not_started", "in_progress"):
                continue
            msgs = sdata.get("messages", [])
            if not msgs:
                continue
            # Try multiple quiz score field names
            quiz = sdata.get("quiz_score") or sdata.get("score")
            total_q = sdata.get("total_questions") or sdata.get("num_questions")
            # Also try computing from results array
            if quiz is None and "quiz_results" in sdata:
                results = sdata["quiz_results"]
                if isinstance(results, list):
                    correct = sum(1 for r in results if r.get("correct") or r.get("is_correct"))
                    total_q = len(results)
                    quiz = correct
            dur = sdata.get("duration_seconds") or sdata.get("duration") or 0
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            rows.append({
                "uid":          uid,
                "session":      sname,
                "condition":    cond,
                "quiz_pct":     (quiz / total_q * 100) if quiz is not None and total_q else None,
                "duration_min": dur / 60 if dur else None,
                "n_user_msgs":  len(user_msgs),
            })
    df = pd.DataFrame(rows)
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Java Tutor Study")
    st.markdown("*Research Dashboard · 2026*")
    st.divider()

    session_filter = st.selectbox(
        "Session subset",
        ["all3", "arraylist_recursion", "arraylist_queue", "recursion_queue"],
        format_func=lambda x: {
            "all3":                  "All 3 sessions",
            "arraylist_recursion":   "ArrayList + Recursion",
            "arraylist_queue":       "ArrayList + Queue",
            "recursion_queue":       "Recursion + Queue",
        }[x]
    )

    st.divider()
    st.markdown("### 🎯 Intervention Simulator")
    st.markdown("*Set thresholds to flag struggling students*")
    thresh_msgs  = st.slider("Min messages before flagging", 1, 10, 3)
    thresh_think = st.slider("Max think time (seconds)", 30, 300, 120)
    thresh_quiz  = st.slider("Quiz score threshold (%)", 30, 70, 60)

    st.divider()
    st.markdown("### 💡 Next Paper Ideas")
    ideas = [
        "Early intervention trigger from behavioral signals",
        "Instructor dashboard for real-time monitoring",
        "Adaptive scaffolding based on engagement level",
        "Longitudinal study across full semester",
        "Cross-topic transfer of scaffolding benefits",
    ]
    idea_votes = {i: st.checkbox(i, key=f"idea_{i[:20]}") for i in ideas}
    if any(idea_votes.values()):
        st.success(f"✓ {sum(idea_votes.values())} idea(s) flagged for next paper")

# ── Load data ─────────────────────────────────────────────────────────────────
clusters    = load_clusters(session_filter)
resp_times  = load_response_times(session_filter)
char_counts = load_char_counts(session_filter)
qq          = load_question_quality(session_filter)
df_sess     = load_session_data()

assignments = pd.DataFrame(clusters["assignments"])
centroids   = pd.DataFrame(clusters["centroids"])
cond_mix    = pd.DataFrame(clusters["condition_mix"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Java Tutor Study")
st.markdown("*Behavioral fingerprints, learning outcomes & intervention signals*")
st.divider()

# ── Key metrics ───────────────────────────────────────────────────────────────
n_students = assignments["uid"].nunique()
n_sessions = len(assignments)
high_pct   = round(len(assignments[assignments["perf_tier"]=="High"]) / n_sessions * 100)
low_pct    = round(len(assignments[assignments["perf_tier"]=="Low"])  / n_sessions * 100)
scaffolded = assignments[assignments["condition"].isin(["character_scaffolded","non_character_scaffolded"])]
direct     = assignments[assignments["condition"] == "direct_chat"]
gap        = round(scaffolded["quiz_pct"].mean() - direct["quiz_pct"].mean(), 1)

c1,c2,c3,c4,c5 = st.columns(5)
for col, val, label in [
    (c1, n_students, "Students"),
    (c2, n_sessions, "Sessions"),
    (c3, f"+{gap}pp", "Scaffolding Advantage"),
    (c4, f"{high_pct}%", "High Performers"),
    (c5, f"{low_pct}%", "At-Risk Students"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-big">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Learning Outcomes",
    "🔬 Behavioral Fingerprints",
    "⏱ Response Patterns",
    "🚨 Intervention Simulator",
    "💬 Message Analysis",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Learning Outcomes
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Learning Outcomes by Condition</div>', unsafe_allow_html=True)

    quiz_by_cond = assignments.groupby("condition")["quiz_pct"].agg(["mean","sem"]).reset_index()

    fig = go.Figure(data=[go.Bar(
        x=[COND_LABELS[c] for c in quiz_by_cond["condition"]],
        y=quiz_by_cond["mean"].tolist(),
        error_y=dict(type='data', array=(quiz_by_cond["sem"]*1.96).tolist(),
                     visible=True, color='rgba(255,255,255,0.5)', thickness=2),
        marker_color=[COND_COLORS[c] for c in quiz_by_cond["condition"]],
        marker_line_width=0,
        text=[f'{v:.1f}%' for v in quiz_by_cond["mean"]],
        textposition='outside',
        textfont=dict(family='DM Mono', size=14, color='white'),
    )])
    fig.update_layout(
        **LAYOUT_BASE,
        title="Mean Quiz Score by Condition",
        yaxis=dict(range=[0,105], gridcolor='rgba(99,102,241,0.1)', ticksuffix='%'),
        showlegend=False, bargap=0.35, height=400,
    )
    st.plotly_chart(fig, width='stretch')

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown('<div class="finding-box">📈 Scaffolded conditions score <strong>15–19 points higher</strong> than Direct Chat (p = 0.007, Cohen\'s d = 0.89)</div>', unsafe_allow_html=True)
    with col_f2:
        st.markdown('<div class="finding-box">⚡ Scaffolded students achieve better results in <strong>half the time</strong> (~9 min vs ~17.5 min)</div>', unsafe_allow_html=True)
    with col_f3:
        st.markdown('<div class="finding-box">❓ More time ≠ better score — Direct Chat students spend 2× longer but score worse</div>', unsafe_allow_html=True)

    # Performance by session topic
    st.markdown('<div class="section-header">Performance by Session Topic</div>', unsafe_allow_html=True)

    if df_sess is not None:
        sess_quiz = df_sess.dropna(subset=["quiz_pct"]).groupby(
            ["session","condition"])["quiz_pct"].mean().reset_index()

        if not sess_quiz.empty:
            # Sort sessions in logical order
            sess_order = ["arraylist", "queue", "recursion"]
            sess_quiz["session"] = pd.Categorical(sess_quiz["session"],
                                                   categories=sess_order, ordered=True)
            sess_quiz = sess_quiz.sort_values("session")

            fig12 = go.Figure()
            for cond in ["character_scaffolded", "non_character_scaffolded", "direct_chat"]:
                grp = sess_quiz[sess_quiz["condition"] == cond]
                fig12.add_trace(go.Bar(
                    name=COND_LABELS[cond],
                    x=grp["session"].astype(str).str.capitalize(),
                    y=grp["quiz_pct"],
                    marker_color=COND_COLORS[cond],
                    marker_line_width=0,
                    text=[f'{v:.0f}%' for v in grp["quiz_pct"]],
                    textposition='outside',
                    textfont=dict(family='DM Mono', size=11, color='white'),
                ))
            fig12.update_layout(
                **LAYOUT_BASE,
                barmode='group',
                title="Quiz Score by Topic and Condition",
                yaxis=dict(gridcolor='rgba(99,102,241,0.1)', ticksuffix='%', range=[0,115]),
                xaxis=dict(title="Topic"),
                legend=dict(bgcolor='rgba(26,29,46,0.8)', bordercolor='rgba(99,102,241,0.3)',
                            borderwidth=1, font=dict(size=11)),
                height=400,
            )
            st.plotly_chart(fig12, width='stretch')
        else:
            st.info("Quiz score data not available in session JSON — scores are in the cluster assignments above.")
    else:
        st.info("Session data file not found.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Behavioral Fingerprints
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Behavioral Fingerprints</div>', unsafe_allow_html=True)

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        k_val = st.slider("Number of clusters (k)", 2, 5, 3)

    if k_val == 3:
        cents   = centroids.copy()
        assigns = assignments.copy()
    else:
        features = ["quiz_pct","mean_depth","pct_conceptual","med_user_len","med_think_sec","n_user_msgs"]
        df_km    = assignments[features].dropna()
        scaler   = StandardScaler()
        X        = scaler.fit_transform(df_km)
        km       = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels   = km.fit_predict(X)
        df_km    = df_km.copy()
        df_km["cluster_id"] = labels
        cents    = df_km.groupby("cluster_id")[features].mean().reset_index()
        tier_names = ["Low","Medium","High","Very High","Elite"]
        cents["perf_tier"] = cents["quiz_pct"].rank(method='first').apply(
            lambda r: tier_names[min(int(r)-1, len(tier_names)-1)]
        )
        assigns = assignments.copy()
        assigns["cluster_id"] = km.predict(scaler.transform(assignments[features].fillna(0)))
        assigns["perf_tier"]  = assigns["cluster_id"].map(
            dict(zip(cents["cluster_id"], cents["perf_tier"]))
        )

    tier_color_map = {'High':'#4ade80','Medium':'#fb923c','Low':'#f87171',
                      'Very High':'#60a5fa','Elite':'#c084fc'}

    col_l2, col_r2 = st.columns([1.2, 0.8])

    with col_l2:
        features_radar = ["quiz_pct","mean_depth","med_user_len","med_think_sec","n_user_msgs"]
        feat_labels_r  = ["Quiz %","Depth Score","Msg Length","Think Time","# Messages"]

        all_vals  = np.array([[cents[cents["perf_tier"]==t][f].values[0]
                               if len(cents[cents["perf_tier"]==t]) else 0
                               for f in features_radar]
                              for t in cents["perf_tier"].unique()])
        col_min   = all_vals.min(0)
        col_max   = all_vals.max(0)
        col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)

        fig3 = go.Figure()
        for _, row in cents.iterrows():
            tier  = row["perf_tier"]
            vals  = [(row[f] - col_min[i]) / col_range[i] for i, f in enumerate(features_radar)]
            color = tier_color_map.get(tier, '#94a3b8')
            r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig3.add_trace(go.Scatterpolar(
                r=vals+[vals[0]],
                theta=feat_labels_r+[feat_labels_r[0]],
                fill='toself',
                fillcolor=f"rgba({r},{g},{b},0.15)",
                line=dict(color=color, width=2),
                name=f"{tier} (quiz: {row['quiz_pct']:.0f}%)",
            ))
        fig3.update_layout(
            **LAYOUT_BASE,
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='rgba(99,102,241,0.2)',
                                tickfont=dict(color='#475569', size=9)),
                angularaxis=dict(tickfont=dict(color='#94a3b8', size=11),
                                 gridcolor='rgba(99,102,241,0.2)'),
            ),
            title=dict(text=f"Behavioral Fingerprints (k={k_val})",
                       font=dict(color='#ffffff', family='DM Serif Display', size=16)),
            legend=dict(bgcolor='rgba(26,29,46,0.8)', bordercolor='rgba(99,102,241,0.3)',
                        borderwidth=1, font=dict(size=11)),
            height=420,
        )
        st.plotly_chart(fig3, width='stretch')

    with col_r2:
        st.markdown("**Cluster Profiles**")
        for _, row in cents.iterrows():
            tier = row["perf_tier"]
            cc   = "high" if "High" in tier or "Elite" in tier else ("low" if "Low" in tier else "medium")
            st.markdown(f"""
            <div style="background:rgba(26,29,46,0.8);border:1px solid rgba(99,102,241,0.2);
                        border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span class="pill pill-{cc}">{tier}</span>
                    <span style="font-family:'DM Mono';font-size:1.1rem;color:{tier_color_map.get(tier,'#94a3b8')}">
                        {row['quiz_pct']:.0f}%
                    </span>
                </div>
                <div style="font-size:0.78rem;color:#94a3b8;line-height:1.8;">
                    💬 {row['n_user_msgs']:.0f} messages &nbsp;|&nbsp; ⏱ {row['med_think_sec']:.0f}s think time<br>
                    📏 {row['med_user_len']:.0f} chars/msg &nbsp;|&nbsp; 🧠 depth: {row['mean_depth']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Condition → Cluster Distribution**")
        mix_pivot = cond_mix.pivot_table(index='perf_tier', columns='condition',
                                          values='count', fill_value=0)
        mix_pivot.columns = [COND_LABELS.get(c,c) for c in mix_pivot.columns]
        st.dataframe(mix_pivot, width='stretch')

    st.markdown('<div class="section-header">Think Time vs Quiz Score</div>', unsafe_allow_html=True)
    fig4 = px.scatter(
        assigns.dropna(subset=["quiz_pct","med_think_sec"]),
        x="med_think_sec", y="quiz_pct",
        color="perf_tier", color_discrete_map=TIER_COLORS,
        symbol="condition",
        hover_data=["uid","session","n_user_msgs"],
        labels={"med_think_sec":"Median Think Time (s)","quiz_pct":"Quiz Score (%)","perf_tier":"Tier"},
        title="The disengagement signal: very long think times = Low performers",
    )
    fig4.update_traces(marker=dict(size=9, opacity=0.8,
                                   line=dict(width=1, color='rgba(255,255,255,0.2)')))
    fig4.update_layout(
        **LAYOUT_BASE,
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)', ticksuffix='%'),
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)', ticksuffix='s'),
        legend=dict(bgcolor='rgba(26,29,46,0.8)', bordercolor='rgba(99,102,241,0.3)', borderwidth=1),
        height=380,
    )
    st.plotly_chart(fig4, width='stretch')
    st.markdown('<div class="finding-box">🔑 <strong>The disengagement pattern:</strong> Low performers have median think times of 581s — they start sessions but barely interact. The intervention point is early: after 2 messages with >120s gaps, the student is likely to disengage entirely.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Response Patterns
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Response Time Patterns</div>', unsafe_allow_html=True)

    tutor_lat     = pd.DataFrame(resp_times["tutor_latency_by_condition"])
    student_think = pd.DataFrame(resp_times["student_think_by_condition"])

    col_l3, col_r3 = st.columns(2)

    with col_l3:
        fig5 = go.Figure(data=[go.Bar(
            x=[COND_LABELS[c] for c in tutor_lat["condition"]],
            y=tutor_lat["mean"].tolist(),
            marker_color=[COND_COLORS[c] for c in tutor_lat["condition"]],
            marker_line_width=0,
            text=[f'{v:.1f}s' for v in tutor_lat["mean"]],
            textposition='outside',
            textfont=dict(family='DM Mono', size=13, color='white'),
        )])
        fig5.update_layout(
            **LAYOUT_BASE,
            title="Tutor Response Latency",
            yaxis=dict(gridcolor='rgba(99,102,241,0.1)', ticksuffix='s'),
            showlegend=False, bargap=0.35, height=350,
        )
        st.plotly_chart(fig5, width='stretch')
        st.markdown('<div class="finding-box">Direct Chat waits <strong>3.5× longer</strong> for a response (8.3s vs 2.4s) — information overload may partially explain the learning gap</div>', unsafe_allow_html=True)

    with col_r3:
        fig6 = go.Figure(data=[go.Bar(
            x=[COND_LABELS[c] for c in student_think["condition"]],
            y=student_think["mean"].tolist(),
            marker_color=[COND_COLORS[c] for c in student_think["condition"]],
            marker_line_width=0,
            text=[f'{v:.0f}s' for v in student_think["mean"]],
            textposition='outside',
            textfont=dict(family='DM Mono', size=13, color='white'),
            error_y=dict(type='data',
                         array=(student_think["std"]/np.sqrt(student_think["n"])).tolist(),
                         visible=True, color='rgba(255,255,255,0.4)'),
        )])
        fig6.update_layout(
            **LAYOUT_BASE,
            title="Student Think Time",
            yaxis=dict(gridcolor='rgba(99,102,241,0.1)', ticksuffix='s'),
            showlegend=False, bargap=0.35, height=350,
        )
        st.plotly_chart(fig6, width='stretch')
        r = resp_times["think_time_quiz_correlation"]["pearson_r"]
        p = resp_times["think_time_quiz_correlation"]["pearson_p"]
        st.markdown(f'<div class="finding-box">Think time ↔ quiz: r = {r:.3f}, p = {p:.3f} — longer think time is a weak negative predictor of performance</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Message Length by Role & Condition</div>', unsafe_allow_html=True)
    char_df = pd.DataFrame(char_counts["by_condition_role"])
    asst    = char_df[char_df["role"]=="assistant"]
    user    = char_df[char_df["role"]=="user"]

    col_c1, col_c2 = st.columns(2)
    for col, df_sub, role in [(col_c1, asst, "Tutor"), (col_c2, user, "Student")]:
        with col:
            fig7 = go.Figure()
            for _, row in df_sub.iterrows():
                cond = row["condition"]
                fig7.add_trace(go.Box(
                    q1=[row["p25"]], median=[row["median"]], q3=[row["p75"]],
                    mean=[row["mean"]],
                    lowerfence=[max(0, row["median"]-row["std"])],
                    upperfence=[row["median"]+row["std"]],
                    name=COND_LABELS[cond],
                    marker_color=COND_COLORS[cond],
                    line_width=2,
                ))
            fig7.update_layout(
                **LAYOUT_BASE,
                title=f"{role} Message Length (chars)",
                yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
                height=320,
            )
            st.plotly_chart(fig7, width='stretch')

    st.markdown('<div class="finding-box">🤯 <strong>Information overload:</strong> Direct Chat tutor responses average 1,705 chars vs ~430 chars in scaffolded conditions — 4× more text per message. Students receiving walls of text score 20 points lower.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Intervention Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🚨 Early Intervention Simulator</div>', unsafe_allow_html=True)
    st.markdown("*Adjust thresholds in the sidebar to see how many at-risk students each rule would catch*")

    struggling     = assignments[assignments["quiz_pct"] <  thresh_quiz].copy()
    not_struggling = assignments[assignments["quiz_pct"] >= thresh_quiz].copy()

    def would_flag(row):
        return (row["n_user_msgs"] <= thresh_msgs) or (row["med_think_sec"] >= thresh_think)

    assignments["flagged"]    = assignments.apply(would_flag, axis=1)
    struggling["flagged"]     = struggling.apply(would_flag, axis=1)
    not_struggling["flagged"] = not_struggling.apply(would_flag, axis=1)

    true_pos      = int(struggling["flagged"].sum())
    false_pos     = int(not_struggling["flagged"].sum())
    false_neg     = int((~struggling["flagged"]).sum())
    true_neg      = int((~not_struggling["flagged"]).sum())
    total_flagged = int(assignments["flagged"].sum())

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall    = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    cm1,cm2,cm3,cm4 = st.columns(4)
    for col, val, label in [
        (cm1, total_flagged, "Sessions Flagged"),
        (cm2, true_pos,      "True Positives"),
        (cm3, false_pos,     "False Alarms"),
        (cm4, f"{f1:.2f}",   "F1 Score"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-big">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_l4, col_r4 = st.columns([1, 1.5])

    with col_l4:
        fig8 = go.Figure(data=go.Heatmap(
            z=[[true_neg, false_pos],[false_neg, true_pos]],
            x=["Not Flagged","Flagged"],
            y=["Not Struggling","Struggling"],
            colorscale=[[0,'#1a1d2e'],[1,'#818cf8']],
            text=[[str(true_neg), str(false_pos)],[str(false_neg), str(true_pos)]],
            texttemplate="%{text}",
            textfont=dict(size=24, family='DM Mono', color='white'),
            showscale=False,
        ))
        fig8.update_layout(**LAYOUT_BASE, title="Confusion Matrix", height=320)
        st.plotly_chart(fig8, width='stretch')

    with col_r4:
        plot_df = assignments.dropna(subset=["n_user_msgs","med_think_sec","quiz_pct"]).copy()
        fig9 = px.scatter(
            plot_df,
            x="n_user_msgs", y="med_think_sec",
            color="flagged",
            color_discrete_map={True:'#f87171', False:'#4ade80'},
            symbol="perf_tier",
            hover_data=["quiz_pct","condition","session"],
            labels={"n_user_msgs":"Messages Sent","med_think_sec":"Think Time (s)","flagged":"Flagged"},
            title="Flagged (red) vs Safe (green)",
        )
        fig9.add_vline(x=thresh_msgs,  line_dash="dash", line_color="#fbbf24",
                       annotation_text=f"≤{thresh_msgs} msgs", annotation_font_color="#fbbf24")
        fig9.add_hline(y=thresh_think, line_dash="dash", line_color="#fbbf24",
                       annotation_text=f"≥{thresh_think}s", annotation_font_color="#fbbf24")
        fig9.update_layout(
            **LAYOUT_BASE,
            yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
            xaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
            legend=dict(bgcolor='rgba(26,29,46,0.8)', font=dict(size=10)),
            height=320,
        )
        st.plotly_chart(fig9, width='stretch')

    st.markdown(f"""
    <div class="finding-box">
    🎯 <strong>Current thresholds:</strong> Flag if ≤{thresh_msgs} messages OR think time ≥{thresh_think}s<br>
    Catches <strong>{recall*100:.0f}%</strong> of struggling students with <strong>{precision*100:.0f}%</strong> precision.
    F1 = {f1:.2f} — {'excellent' if f1>0.7 else 'good' if f1>0.5 else 'moderate'} for an early warning system.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="finding-box">💡 <strong>Next paper hook:</strong> Two behavioral signals (message count + think time) can identify struggling students within the first few minutes — before quiz scores are available.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Message Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Message Depth & Question Quality</div>', unsafe_allow_html=True)

    qq_sess = pd.DataFrame(qq["per_session"])
    qq_corr = qq["correlation"]

    col_l5, col_r5 = st.columns([1.3, 0.7])

    with col_l5:
        qq_by_cond = qq_sess.groupby("condition")[
            ["pct_conceptual","pct_procedural","pct_surface","pct_offtask"]
        ].mean().reset_index()

        fig10 = go.Figure()
        for col_name, label, color in [
            ("pct_conceptual", "Conceptual",  "#818cf8"),
            ("pct_procedural", "Procedural",  "#34d399"),
            ("pct_surface",    "Surface",     "#fbbf24"),
            ("pct_offtask",    "Off-task/Ack","#475569"),
        ]:
            fig10.add_trace(go.Bar(
                name=label,
                x=[COND_LABELS[c] for c in qq_by_cond["condition"]],
                y=qq_by_cond[col_name],
                marker_color=color,
                marker_line_width=0,
            ))
        fig10.update_layout(
            **LAYOUT_BASE,
            barmode='stack',
            title="Message Type Distribution by Condition",
            yaxis=dict(gridcolor='rgba(99,102,241,0.1)', ticksuffix='%'),
            legend=dict(bgcolor='rgba(26,29,46,0.8)', font=dict(size=11)),
            height=380,
        )
        st.plotly_chart(fig10, width='stretch')

    with col_r5:
        d    = qq_sess.dropna(subset=["quiz_pct","mean_depth"])
        fig11 = px.scatter(
            d, x="mean_depth", y="quiz_pct",
            color="condition", color_discrete_map=COND_COLORS,
            hover_data=["session","n_msgs"],
            labels={"mean_depth":"Mean Depth Score","quiz_pct":"Quiz Score (%)"},
            title=f"Depth vs Quiz  r={qq_corr.get('pearson_r',0):.3f}  p={qq_corr.get('pearson_p',0):.3f}",
        )
        m, b = np.polyfit(d["mean_depth"], d["quiz_pct"], 1)
        xs   = np.linspace(d["mean_depth"].min(), d["mean_depth"].max(), 100)
        fig11.add_trace(go.Scatter(x=xs, y=m*xs+b, mode='lines',
                                   line=dict(color='white', dash='dash', width=1.5),
                                   showlegend=False))
        fig11.update_layout(
            **LAYOUT_BASE,
            yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
            xaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
            height=380,
        )
        st.plotly_chart(fig11, width='stretch')

    col_f5a, col_f5b = st.columns(2)
    with col_f5a:
        st.markdown('<div class="finding-box">🤔 <strong>Counterintuitive:</strong> Deeper questions are negatively correlated with quiz performance. Students asking the most conceptual questions are struggling — not thriving.</div>', unsafe_allow_html=True)
    with col_f5b:
        st.markdown('<div class="finding-box">✅ <strong>Acknowledgments ARE engagement:</strong> ~88% of scaffolded messages are short affirmations. This is correct behavior in a guided flow.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.75rem;font-family:'DM Mono',monospace;">
Java Tutor Study · College of Charleston · 2026 · For research team use only
</div>
""", unsafe_allow_html=True)