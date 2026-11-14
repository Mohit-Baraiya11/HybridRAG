"""
GraphAgenticRAG — Latency Analytics Dashboard
Run: uv run streamlit run dashboard.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphAgenticRAG · Latency Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
  }
  div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
  }
  div[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
  }
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin: 28px 0 12px;
  }
  .hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
  }
  .hero-sub { color: rgba(255,255,255,0.4); font-size: 0.85rem; margin-bottom: 28px; }
</style>
""", unsafe_allow_html=True)

# ── Supabase (service role → all users) ──────────────────────────────────────
@st.cache_resource
def get_supabase():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

sb = get_supabase()

def percentile(series, p):
    if len(series) == 0:
        return 0.0
    return round(float(np.percentile(series.dropna(), p)), 2)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_latency_logs(days=30):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        res = sb.table("latency_logs") \
            .select("id,user_id,session_id,latency_seconds,routed_tools,tokens_used,created_at") \
            .gte("created_at", since) \
            .order("created_at", desc=True) \
            .limit(5000) \
            .execute()
        df = pd.DataFrame(res.data)
    except Exception as e:
        st.warning(f"latency_logs table not found or empty — send a few messages first to generate data. ({e})")
        return pd.DataFrame()

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce")
    df["tokens_used"] = pd.to_numeric(df["tokens_used"], errors="coerce").fillna(0).astype(int)
    df["tools_str"] = df["routed_tools"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) and x else "direct"
    )
    df["date"] = df["created_at"].dt.date
    df["hour_of_day"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.day_name()
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ GraphAgenticRAG")
    st.markdown("<div class='section-header'>Filters</div>", unsafe_allow_html=True)
    time_window = st.selectbox("Time window", [1, 7, 14, 30, 90], index=2,
                                format_func=lambda d: f"Last {d} day{'s' if d>1 else ''}")
    st.markdown("<div class='section-header'>Options</div>", unsafe_allow_html=True)
    show_raw = st.checkbox("Show raw data table", value=False)
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
    st.markdown("---")
    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("P50 = median latency\nP95 = 95th percentile\nAll times in seconds.")

if auto_refresh:
    import time as _t; _t.sleep(30)
    st.cache_data.clear(); st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_latency_logs(days=time_window)

st.markdown("<div class='hero-title'>⚡ Latency Analytics</div>", unsafe_allow_html=True)
st.markdown(f"<div class='hero-sub'>GraphAgenticRAG · All Users · Last {time_window} day{'s' if time_window>1 else ''}</div>",
            unsafe_allow_html=True)

if df.empty:
    st.info("📭 No latency data yet. Start chatting at http://127.0.0.1:8000 and data will appear here automatically.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
p50 = percentile(df["latency_seconds"], 50)
p95 = percentile(df["latency_seconds"], 95)
p99 = percentile(df["latency_seconds"], 99)
mean_lat = round(df["latency_seconds"].mean(), 2)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("P50 Latency",    f"{p50}s",  help="Median response time across all users")
c2.metric("P95 Latency",    f"{p95}s",  help="95th percentile — tail latency")
c3.metric("P99 Latency",    f"{p99}s",  help="99th percentile — worst case")
c4.metric("Mean Latency",   f"{mean_lat}s")
c5.metric("Total Requests", f"{len(df):,}")
c6.metric("Unique Users",   f"{df['user_id'].nunique():,}")

st.markdown("---")

TOOL_COLORS = {
    "hybrid_rag": "#818cf8", "graph_rag": "#c084fc",
    "web_search": "#34d399", "github_rag": "#fb923c", "direct": "#94a3b8",
}

# ── Row 1: Histogram + Tool P50/P95 bar ──────────────────────────────────────
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown("<div class='section-header'>Latency Distribution — all requests</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["latency_seconds"], nbinsx=40,
        marker=dict(color="rgba(129,140,248,0.7)", line=dict(color="rgba(129,140,248,1)", width=1)),
        hovertemplate="Latency: %{x:.2f}s<br>Count: %{y}<extra></extra>",
    ))
    for val, lbl, col in [(p50, "P50", "#34d399"), (p95, "P95", "#f87171")]:
        fig.add_vline(x=val, line_dash="dash", line_color=col, line_width=2,
                      annotation_text=f" {lbl}={val}s", annotation_font_color=col,
                      annotation_position="top right")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                      font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
                      margin=dict(l=0,r=0,t=10,b=0), height=280, showlegend=False,
                      xaxis=dict(title="Latency (s)", gridcolor="rgba(255,255,255,0.07)"),
                      yaxis=dict(title="Requests",    gridcolor="rgba(255,255,255,0.07)"))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("<div class='section-header'>P50 vs P95 by Tool</div>", unsafe_allow_html=True)
    tool_rows = []
    for _, row in df.iterrows():
        tools = row["routed_tools"] if isinstance(row["routed_tools"], list) and row["routed_tools"] else ["direct"]
        for t in tools:
            tool_rows.append({"tool": t, "latency_seconds": row["latency_seconds"]})
    tool_df = pd.DataFrame(tool_rows)

    tool_stats = tool_df.groupby("tool")["latency_seconds"].agg(
        p50=lambda x: percentile(x, 50),
        p95=lambda x: percentile(x, 95),
    ).reset_index().sort_values("p95")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=tool_stats["tool"], x=tool_stats["p50"], name="P50",
                          orientation="h", opacity=0.6,
                          marker_color=[TOOL_COLORS.get(t, "#818cf8") for t in tool_stats["tool"]],
                          hovertemplate="%{y}: P50=%{x:.2f}s<extra></extra>"))
    fig2.add_trace(go.Bar(y=tool_stats["tool"], x=tool_stats["p95"], name="P95",
                          orientation="h", opacity=1.0,
                          marker_color=[TOOL_COLORS.get(t, "#f87171") for t in tool_stats["tool"]],
                          hovertemplate="%{y}: P95=%{x:.2f}s<extra></extra>"))
    fig2.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(255,255,255,0.03)",
                       font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
                       margin=dict(l=0,r=0,t=10,b=0), height=280,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                       xaxis=dict(title="Latency (s)", gridcolor="rgba(255,255,255,0.07)"),
                       yaxis=dict(gridcolor="rgba(255,255,255,0.07)"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Trend ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>P50 & P95 Trend Over Time</div>", unsafe_allow_html=True)
daily = df.groupby("date")["latency_seconds"].agg(
    p50=lambda x: percentile(x, 50),
    p95=lambda x: percentile(x, 95),
    count="count"
).reset_index()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=daily["date"], y=daily["p95"], name="P95",
                           mode="lines+markers", line=dict(color="#f87171", width=2.5),
                           fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
                           hovertemplate="Date: %{x}<br>P95: %{y:.2f}s<extra></extra>"))
fig3.add_trace(go.Scatter(x=daily["date"], y=daily["p50"], name="P50",
                           mode="lines+markers", line=dict(color="#34d399", width=2.5),
                           fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
                           hovertemplate="Date: %{x}<br>P50: %{y:.2f}s<extra></extra>"))
fig3.add_trace(go.Bar(x=daily["date"], y=daily["count"], name="Requests", yaxis="y2",
                       marker_color="rgba(129,140,248,0.2)",
                       hovertemplate="Date: %{x}<br>Requests: %{y}<extra></extra>"))
fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
                    margin=dict(l=0,r=0,t=10,b=0), height=280, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                    yaxis=dict(title="Latency (s)", gridcolor="rgba(255,255,255,0.07)"),
                    yaxis2=dict(title="Requests", overlaying="y", side="right", showgrid=False))
st.plotly_chart(fig3, use_container_width=True)

# ── Row 3: Per-user table + tool pie ─────────────────────────────────────────
st.markdown("---")
ca, cb = st.columns([2, 3])

with ca:
    st.markdown("<div class='section-header'>Per-User P50 / P95</div>", unsafe_allow_html=True)
    user_stats = df.groupby("user_id")["latency_seconds"].agg(
        P50=lambda x: percentile(x, 50),
        P95=lambda x: percentile(x, 95),
        Requests="count",
    ).reset_index().sort_values("P95", ascending=False)
    user_stats["User"] = user_stats["user_id"].str[:10] + "…"
    st.dataframe(user_stats[["User","P50","P95","Requests"]], use_container_width=True,
                 hide_index=True,
                 column_config={"P50": st.column_config.NumberColumn(format="%.2f s"),
                                "P95": st.column_config.NumberColumn(format="%.2f s")})

with cb:
    st.markdown("<div class='section-header'>Tool Usage Distribution</div>", unsafe_allow_html=True)
    tc = tool_df["tool"].value_counts().reset_index()
    tc.columns = ["Tool","Count"]
    fig4 = go.Figure(go.Pie(
        labels=tc["Tool"], values=tc["Count"], hole=0.52,
        marker=dict(colors=[TOOL_COLORS.get(t,"#64748b") for t in tc["Tool"]]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} requests (%{percent})<extra></extra>",
    ))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="rgba(255,255,255,0.75)"),
                        margin=dict(l=0,r=0,t=10,b=0), height=300,
                        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig4, use_container_width=True)

# ── Detailed tool table ───────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Full Tool Latency Breakdown</div>", unsafe_allow_html=True)
tbl = tool_df.groupby("tool")["latency_seconds"].agg(
    P50=lambda x: percentile(x, 50),
    P75=lambda x: percentile(x, 75),
    P95=lambda x: percentile(x, 95),
    P99=lambda x: percentile(x, 99),
    Mean=lambda x: round(x.mean(), 2),
    Requests="count",
).reset_index().rename(columns={"tool":"Tool"}).sort_values("P95", ascending=False)
st.dataframe(tbl, use_container_width=True, hide_index=True,
             column_config={c: st.column_config.NumberColumn(format="%.2f s")
                            for c in ["P50","P75","P95","P99","Mean"]})

# ── Hourly heatmap ────────────────────────────────────────────────────────────
if df["date"].nunique() > 2:
    st.markdown("<div class='section-header'>Hourly P95 Heatmap (by Day × Hour)</div>", unsafe_allow_html=True)
    heat = df.groupby(["day_of_week","hour_of_day"])["latency_seconds"] \
             .apply(lambda x: percentile(x, 95)).reset_index()
    heat.columns = ["Day","Hour","P95"]
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = heat.pivot(index="Day", columns="Hour", values="P95").reindex(day_order)
    fig5 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale=[[0,"#1e1b4b"],[0.5,"#818cf8"],[1,"#f87171"]],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>P95: %{z:.2f}s<extra></extra>",
        colorbar=dict(title="P95 (s)", tickfont=dict(color="rgba(255,255,255,0.6)")),
    ))
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                        font=dict(family="Inter", color="rgba(255,255,255,0.7)"),
                        margin=dict(l=0,r=0,t=10,b=0), height=260,
                        xaxis=dict(title="Hour of Day"), yaxis=dict(title=""))
    st.plotly_chart(fig5, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown("<div class='section-header'>Raw Latency Log</div>", unsafe_allow_html=True)
    disp = df[["created_at","user_id","session_id","latency_seconds","tools_str","tokens_used"]].copy()
    disp["user_id"] = disp["user_id"].str[:12] + "…"
    disp["session_id"] = disp["session_id"].str[:8] + "…"
    st.dataframe(disp.rename(columns={"created_at":"Timestamp","user_id":"User","session_id":"Session",
                                       "latency_seconds":"Latency (s)","tools_str":"Tools","tokens_used":"Tokens"}),
                 use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')} · {len(df):,} rows · window: {time_window}d")
