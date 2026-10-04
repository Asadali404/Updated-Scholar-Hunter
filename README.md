# ScholarHunter Agents

Multi-agent scholarship finder built with **CrewAI**, **Groq (`openai/gpt-oss-120b` only)** and **Streamlit**.
All tools are free: pypdf, DuckDuckGo search (ddgs), requests, pandas/openpyxl.

## Agents
1. **Profile Analyzer** – CV (first 3,500 chars) to profile + keywords
2. **Web Search Specialist** – plans queries, runs DuckDuckGo (max 4 queries x 4 results), fetches pages
3. **Database & Skill Gap Mentor** – raw results to records, then gap report
4. **Verification Officer** – gatekeeper: link must open, domain must be official/academic, LLM must confirm it is a genuine programme, deadline is read from the page by code (never guessed), expired items are dropped
5. **Tracker & Deadline Coordinator** – urgency, metrics, optional one-paragraph briefing

Only verified records are shown. If nothing passes, a curated list is shown and labelled *From curated list, verify*.

## Deploy (Streamlit Community Cloud)
1. Push this folder to GitHub.
2. Create the app from `app.py`; in **Advanced settings** pick **Python 3.11**.
3. Paste in the Secrets panel: `GROQ_API_KEY = "your_key"` (free key from console.groq.com).
4. If the build fails on sqlite3, `pysqlite3-binary` plus the override at the top of `app.py` already handles it.

## Pin versions (recommended)
After the first successful run in a Python 3.11 venv: `pip freeze | grep -iE "streamlit|crewai|litellm|pydantic|pandas|openpyxl|pypdf|requests|ddgs"` and copy the exact versions into `requirements.txt`.

## Local run
`pip install -r requirements.txt && streamlit run app.py` and `pytest -q` for tests.
Key order: sidebar key, `st.secrets`, then `GROQ_API_KEY` env var. Never commit `.streamlit/secrets.toml` or `.env`.
