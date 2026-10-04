# --- Streamlit Cloud sqlite fix (CrewAI/chromadb need sqlite >= 3.35). Must stay at the very top. ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import pandas as pd
import streamlit as st

import tracker as tk
from crew import run_pipeline
from export import records_df, to_excel
from tools import pdf_extract

st.set_page_config(page_title="ScholarHunter Agents", page_icon="🎓", layout="wide")

PRESETS = ["USA", "UK", "Germany", "Canada", "Australia", "Japan", "China", "South Korea",
           "Turkey", "Hungary", "Netherlands", "Sweden", "France", "Italy", "Malaysia"]


def get_api_key(sidebar_key: str) -> str:
    """Read order: sidebar personal key, st.secrets, environment. Never printed."""
    if sidebar_key.strip():
        return sidebar_key.strip()
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 ScholarHunter")
    cv_file = st.file_uploader("CV (PDF)", type=["pdf"])
    cv_paste = st.text_area("...or paste CV text (for scanned PDFs)", height=100)
    interests = st.text_area("Research interests", placeholder="e.g. machine learning, NLP, healthcare AI")
    domain = st.text_input("Target domain (optional)")
    level = st.selectbox("Level", ["MS", "PhD", "Any"])
    countries = st.multiselect("Countries", PRESETS, default=["Germany", "UK", "Turkey"])
    other = st.text_input("Other countries (comma-separated)")
    mode = st.radio("Mode", ["Demo", "Live", "Lite"], index=0,
                    help="Demo: sample data, no key. Live: full agents. Lite: fewer searches, faster.")
    personal_key = st.text_input("Personal Groq key (optional)", type="password")
    run = st.button("🚀 Run Agents", type="primary", use_container_width=True)

api_key = get_api_key(personal_key)
all_countries = countries + [c.strip() for c in other.split(",") if c.strip()]
effective_mode = mode
if mode in ("Live", "Lite") and not api_key:
    effective_mode = "Demo"

# ----------------------------------------------------------------- banner
st.title("ScholarHunter Agents")
if mode in ("Live", "Lite") and not api_key:
    st.warning("No Groq API key found (sidebar, Streamlit secrets or environment). Switched to Demo mode automatically.")
elif effective_mode == "Demo":
    st.info("Demo mode: sample data from a curated list. Nothing here is live-verified.")
else:
    st.success(f"{effective_mode} mode: agents run on Groq (openai/gpt-oss-120b). Only verified results are shown.")

# ----------------------------------------------------------------- run
if run:
    cv_text = cv_paste.strip()[:3500]
    if cv_file is not None:
        try:
            cv_text = pdf_extract(cv_file.getvalue())
        except Exception:
            cv_text = ""
        if not cv_text and not cv_paste.strip():
            st.error("No text found in this PDF (it may be scanned). Upload a text-based PDF or paste your CV text in the sidebar.")
            st.stop()
        if not cv_text:
            cv_text = cv_paste.strip()[:3500]
    if effective_mode != "Demo" and not (cv_text or interests.strip()):
        st.error("Please upload a CV, paste CV text, or enter research interests.")
        st.stop()

    with st.status("Agents working...", expanded=True) as status:
        def emit(stage, state, msg):
            icon = {"start": "⏳", "done": "✅", "error": "❌", "info": "•"}.get(state, "•")
            st.write(f"{icon} **{stage}**: {msg}")
        result = run_pipeline(cv_text, interests, domain, level, all_countries, effective_mode, api_key, emit)
        status.update(label="Finished with errors" if result.error else "Finished",
                      state="error" if result.error else "complete")
    st.session_state["result"] = result            # session_state keeps results across reruns/downloads
    st.session_state["tracker_df"] = tk.build_tracker_df(result.records)

result = st.session_state.get("result")
if result is not None and not hasattr(result, "diagnostics"):
    # Stale object saved by an older version of the app: discard it and ask for a fresh run.
    st.session_state.pop("result", None)
    st.session_state.pop("tracker_df", None)
    result = None
if result is None:
    st.write("Fill in the sidebar and press **Run Agents**. Start with Demo mode to see the layout.")
    st.stop()

if result.error:
    st.error(f"Stage '{result.failed_stage}' failed: {result.error} Completed stages are shown below.")
    if result.rate_limited:
        st.info("Options: wait a minute and retry, switch to **Lite** mode, use **Demo** mode, or add your own Groq key in the sidebar.")
for w in result.warnings:
    st.warning(w)
if result.diagnostics:
    with st.expander("Pipeline diagnostics"):
        for d in result.diagnostics:
            st.write("• " + d)

tab1, tab2, tab3, tab4 = st.tabs(["1 Profile & Keywords", "2 Opportunities", "3 Skill Gap & Prep", "4 Tracker & Deadlines"])

with tab1:
    p = result.profile
    if p:
        c1, c2, c3 = st.columns(3)
        c1.metric("Degree", p.degree)
        c2.metric("CGPA", p.cgpa)
        c3.metric("Publications", p.publications)
        st.write(p.summary)
        with st.expander("Skills", expanded=True):
            st.write(", ".join(p.skills) or "None found")
        with st.expander("Research pillars"):
            st.write(", ".join(p.research_areas) or "None found")
        st.markdown("**Keywords:** " + ", ".join(p.keywords))
    else:
        st.info("Profile not available.")

with tab2:
    if result.records:
        df = records_df(result.records)
        n_web = sum(r.source == "web" for r in result.records)
        st.caption(f"{n_web} from live web search, {len(result.records) - n_web} from the curated list.")
        st.dataframe(
            df[["Scholarship", "Country", "Level", "Deadline", "Urgency", "Fit score", "Confidence", "Source", "Verified",
                "Verification note", "Official link"]],
            column_config={
                "Official link": st.column_config.LinkColumn("Official link", display_text="Open"),
                "Fit score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
            }, hide_index=True, use_container_width=True)
    else:
        st.info("No opportunities available.")
    if result.rejected:
        with st.expander(f"Rejected by the Verification Agent ({len(result.rejected)})"):
            st.dataframe(pd.DataFrame(result.rejected), hide_index=True, use_container_width=True)
    st.caption("Verify deadlines on the official site before applying.")

with tab3:
    g = result.gap_report
    if g:
        st.subheader("Overall strengths")
        for s in g.strengths:
            st.markdown(f"- {s}")
        st.subheader("Common gaps")
        for s in g.common_gaps:
            st.markdown(f"- {s}")
        for it in g.items:
            with st.expander(f"{it.scholarship}  |  Priority: {it.priority}  |  Prep: {it.prep_time}"):
                st.markdown("**Weaknesses**\n" + "\n".join(f"- {w}" for w in it.weaknesses))
                st.markdown("**Recommendations**\n" + "\n".join(f"- {w}" for w in it.recommendations))
    else:
        st.info("Gap analysis not available.")

with tab4:
    tdf = st.session_state.get("tracker_df", pd.DataFrame())
    if tdf.empty:
        st.info("Nothing to track yet.")
    else:
        m = tk.metrics(tdf)
        cols = st.columns(5)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, v)
        st.progress(m["Applied"] / m["Total"] if m["Total"] else 0.0, text="Applications submitted")
        edited = st.data_editor(
            tdf, hide_index=True, use_container_width=True, key="tracker_editor",
            disabled=["ID", "Scholarship", "Country", "Deadline", "Days left", "Urgency"],
            column_config={"Status": st.column_config.SelectboxColumn("Status", options=tk.STATUSES),
                           "Notes": st.column_config.TextColumn("Notes")})
        st.session_state["tracker_df"] = edited
        st.subheader("Critical deadlines (next 30 days)")
        crit = tk.critical_list(edited)
        if crit.empty:
            st.caption("No deadlines in the next 30 days.")
        else:
            st.dataframe(crit, hide_index=True, use_container_width=True)
    if result.tracker_summary:
        st.info(result.tracker_summary)

st.divider()
st.download_button("📥 Download Excel", data=to_excel(result, st.session_state.get("tracker_df")),
                   file_name="scholarhunter_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption("Always verify scholarship details and deadlines on the official website. This tool does not guarantee eligibility or acceptance.")
