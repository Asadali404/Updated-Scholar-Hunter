"""Pipeline: 5 CrewAI agents on Groq (openai/gpt-oss-120b only) + free tools.

Agent 1 Profile Analyzer  -> Agent 2 Web Search Specialist -> Agent 3 Database & Skill Gap Mentor
-> Agent 4 Verification Officer (gatekeeper) -> Agent 5 Tracker & Deadline Coordinator
"""
import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from schemas import (CandidateProfile, GapItem, GapReport, PipelineResult,
                     RawResult, ScholarshipRecord)
from tools import (ddg_search, extract_deadline, extract_json, fetch_many, is_aggregator,
                   is_trusted)
import tracker as tk

MODEL = "groq/openai/gpt-oss-120b"   # the ONLY model this app uses
SEED_FILE = Path(__file__).parent / "data" / "seed_scholarships.json"
EU_COUNTRIES = {"Germany", "Netherlands", "Sweden", "France", "Italy", "Hungary"}
Event = Callable[[str, str, str], None]   # (stage, state, message)


class RateLimited(Exception):
    pass


# ------------------------------------------------------------------ LLM helpers
def make_llm(api_key: str):
    from crewai import LLM
    kwargs = dict(model=MODEL, api_key=api_key, temperature=0.1, max_tokens=6000)
    try:
        return LLM(reasoning_effort="low", **kwargs)
    except TypeError:
        return LLM(**kwargs)


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "rate limit" in s or "ratelimit" in s or "rate_limit" in s


AGENTS = {
    "profile": ("Profile Analyzer & Keyword Extractor",
                "Turn a CV into a precise structured profile and search keywords.",
                "You read CVs carefully and never invent facts that are not in the text."),
    "search": ("Scholarship Web Search Specialist",
               "Write focused queries that surface OFFICIAL scholarship pages with deadlines.",
               "You know scholarship portals, ministries and university funding pages."),
    "database": ("Database & Skill Gap Mentor",
                 "Convert raw web evidence into clean records and mentor the candidate on gaps.",
                 "You only use facts present in the supplied evidence and write 'unknown' otherwise."),
    "verify": ("Verification Officer",
               "Reject anything that is not a genuine, official, currently open scholarship.",
               "You are strict and sceptical. When evidence is missing you reject."),
    "tracker": ("Application Tracker & Deadline Coordinator",
                "Summarise the application pipeline and urgent deadlines.",
                "You write short, practical status briefings."),
}


NO_TOOLS = ("\n\nIMPORTANT: there are NO tools or functions available. Never call a tool or function "
            "(not even one named 'Answer' or 'json'). Reply with plain text containing only the requested JSON.")


def _salvage(err: str) -> Optional[str]:
    """Groq 'tool_use_failed' errors embed what the model tried to say; recover JSON if complete."""
    marker = '"failed_generation":"'
    i = err.find(marker)
    if i == -1:
        return None
    frag = err[i + len(marker):].replace('\\"', '"').replace("\\n", "\n")
    try:
        obj = extract_json(frag)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("arguments"), dict):
        obj = obj["arguments"]
    return json.dumps(obj) if isinstance(obj, (dict, list)) else None


def run_agent(key: str, prompt: str, expected: str, llm, retries: int = 3) -> str:
    """Run one CrewAI agent/task with backoff on Groq 429s and retry on tool_use_failed."""
    from crewai import Agent, Crew, Process, Task
    role, goal, backstory = AGENTS[key]
    prompt = prompt + NO_TOOLS
    delay = 15
    for attempt in range(retries + 1):
        try:
            agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm,
                          allow_delegation=False, verbose=False)
            task = Task(description=prompt, expected_output=expected, agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            out = crew.kickoff()
            return str(getattr(out, "raw", out))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if _is_rate_limit(e):
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RateLimited(msg) from e
            if "tool_use_failed" in msg or "Tool choice is none" in msg:
                saved = _salvage(msg)
                if saved:
                    return saved
                if attempt < retries:
                    time.sleep(2)
                    continue
            raise
    raise RateLimited("rate limit")


def call_json(key: str, prompt: str, expected: str, llm, build: Callable, warnings: list):
    """JSON call: retry once with the validation error, then tolerant extract, else None."""
    text = run_agent(key, prompt, expected, llm)
    try:
        return build(extract_json(text))
    except Exception as e1:  # noqa: BLE001
        text = run_agent(key, f"{prompt}\n\nYour previous answer was invalid ({e1}). "
                              f"Return ONLY valid JSON, no markdown.", expected, llm)
        try:
            return build(extract_json(text))
        except Exception as e2:  # noqa: BLE001
            warnings.append(f"{AGENTS[key][0]}: could not parse model output ({e2}).")
            return None


def safe_json(key, prompt, expected, llm, build, warnings):
    """call_json that never aborts the pipeline (except for rate limits)."""
    try:
        return call_json(key, prompt, expected, llm, build, warnings)
    except RateLimited:
        raise
    except Exception as e:  # noqa: BLE001
        warnings.append(f"{AGENTS[key][0]} failed ({type(e).__name__}); continuing with a fallback.")
        return None


def _sleep(seconds: int, emit: Event, stage: str):
    if seconds > 0:
        emit(stage, "info", f"Cool-down {seconds}s to respect Groq limits...")
        time.sleep(seconds)


# ------------------------------------------------------------------ seed / demo
def load_seed(countries: list[str], level: str) -> list[dict]:
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    out = []
    for s in data:
        c_ok = (not countries or s["country"] in countries or
                (s["country"] == "Multiple (EU)" and EU_COUNTRIES & set(countries)))
        l_ok = level == "Any" or level in s["level"]
        if c_ok and l_ok:
            out.append(s)
    return out


def next_occurrence(mm_dd: str, today: Optional[date] = None) -> str:
    today = today or date.today()
    m, d = map(int, mm_dd.split("-"))
    cand = date(today.year, m, d)
    if cand < today:
        cand = date(today.year + 1, m, d)
    return cand.isoformat()


def seed_records(countries, level, label="curated", limit=10) -> list[ScholarshipRecord]:
    recs = []
    for s in load_seed(countries, level)[:limit]:
        dl = next_occurrence(s["typical_deadline"])
        recs.append(ScholarshipRecord(
            name=s["name"], provider=s["provider"], country=s["country"], level=s["level"],
            funding=s["funding"], eligibility=s["eligibility"], deadline=dl,
            days_left=tk.days_left(dl), official_link=s["official_link"], source=label,
            verified=False, confidence="Estimated",
            verification_note="From curated list, verify. Deadline is the typical yearly date, not confirmed."))
    return recs


DEMO_PROFILE = CandidateProfile(
    degree="BS Computer Science", cgpa="3.5/4.0", publications=1,
    skills=["Python", "Machine Learning", "Data Analysis", "SQL"],
    research_areas=["Machine Learning", "Natural Language Processing"],
    keywords=["machine learning", "NLP", "artificial intelligence", "data science", "fully funded", "MS"],
    summary="Sample profile for demo mode. Upload a CV and use Live mode for your own results.")


def demo_result(countries, level) -> PipelineResult:
    countries = countries or ["Germany", "UK", "Turkey", "Hungary", "China"]
    recs = seed_records(countries, level, label="demo")
    for r in recs:
        r.fit_score = score_fit(DEMO_PROFILE, r)
        r.verification_note = "Demo sample from curated list, verify on the official site."
    gap = GapReport(
        strengths=["Solid technical base in Python and ML", "Relevant research direction"],
        common_gaps=["Limited publication record", "No documented IELTS/TOEFL score", "Needs a tailored research proposal"],
        items=[GapItem(scholarship=r.name, weaknesses=["Strengthen research statement", "Add language test score"],
                       recommendations=["Contact a potential supervisor", "Prepare two recommendation letters"],
                       priority="High" if (r.days_left or 99) <= 60 else "Medium", prep_time="4 to 8 weeks")
               for r in recs[:5]])
    attach_gaps(recs, gap)
    return PipelineResult(mode="Demo", profile=DEMO_PROFILE, records=recs, gap_report=gap,
                          tracker_summary="Demo run: sample records only. Switch to Live mode for verified results.",
                          warnings=["Demo mode: records come from a curated list and are not live-verified."])


# ------------------------------------------------------------------ merge helpers
def score_fit(profile: CandidateProfile, rec: ScholarshipRecord) -> int:
    terms = {t.lower() for t in profile.keywords + profile.research_areas + profile.skills if len(t) > 2}
    text = f"{rec.name} {rec.eligibility} {rec.funding} {rec.provider}".lower()
    overlap = sum(1 for t in terms if t in text)
    score = 55 + 7 * overlap
    if "fully" in text:
        score += 5
    return max(40, min(95, score))


def attach_gaps(records, report: Optional[GapReport]):
    if not report:
        return
    for it in report.items:
        for r in records:
            if it.scholarship.lower() in r.name.lower() or r.name.lower() in it.scholarship.lower():
                r.weaknesses = it.weaknesses


# ------------------------------------------------------------------ prompts
def _profile_prompt(cv: str, interests: str, domain: str, level: str) -> str:
    return f"""Analyze this candidate. Use ONLY information found in the CV or interests; write "Unknown" when absent.
CV TEXT:
{cv}
INTERESTS: {interests}
TARGET DOMAIN: {domain or 'not given'}
TARGET LEVEL: {level}
Return ONLY JSON: {{"degree": str, "cgpa": str, "publications": int, "skills": [str], "research_areas": [str],
"keywords": [10-15 short scholarship search keywords], "summary": "2 sentences"}}"""


def _build_profile(d) -> CandidateProfile:
    if isinstance(d, dict) and "publications" in d:
        try:
            d["publications"] = int(re.sub(r"\D", "", str(d["publications"])) or 0)
        except Exception:
            d["publications"] = 0
    return CandidateProfile(**d)


SITE_HINTS = {
    "USA": "edu", "UK": "ac.uk", "Germany": "daad.de", "Canada": "gc.ca", "Australia": "gov.au",
    "Japan": "go.jp", "China": "edu.cn", "South Korea": "go.kr", "Turkey": "gov.tr", "Hungary": "hu",
    "Netherlands": "studyinnl.org", "Sweden": "si.se", "France": "campusfrance.org", "Italy": "esteri.it",
    "Malaysia": "gov.my",
}


def _site_queries(countries, level, year, domain):
    """Deterministic queries restricted to official domains, so results are not blog/aggregator pages."""
    lv = "master's" if level == "MS" else "PhD" if level == "PhD" else "master's PhD"
    if not countries:
        return [f"site:edu {lv} scholarship international students {year} application deadline {domain}".strip()]
    out = []
    for c in countries[:6]:
        site = SITE_HINTS.get(c)
        pre = f"site:{site} " if site else ""
        out.append(f"{pre}{lv} scholarship international students {c} {year} application deadline {domain}".strip())
    return out


def _fallback_queries(countries, level, year, domain):
    lv = "master's" if level == "MS" else "PhD" if level == "PhD" else "master's PhD"
    cs = countries[:4] or ["Europe"]
    return [f"fully funded {lv} scholarship {c} international students apply deadline {year} {domain}".strip()
            for c in cs]


# ------------------------------------------------------------------ main pipeline
def run_pipeline(cv_text: str, interests: str, domain: str, level: str, countries: list[str],
                 mode: str, api_key: str, emit: Event) -> PipelineResult:
    if mode == "Demo" or not api_key:
        emit("demo", "done", "Loaded demo data.")
        return demo_result(countries, level)

    lite = mode == "Lite"
    cooldown = 4 if lite else 15
    max_queries, per_query = (2, 3) if lite else (4, 4)
    today = date.today()
    res = PipelineResult(mode=mode)
    stage = "profile"
    try:
        llm = make_llm(api_key)

        # ---- Agent 1: profile
        emit("Agent 1: Profile Analyzer", "start", "Reading CV and extracting keywords...")
        prof = safe_json("profile", _profile_prompt(cv_text, interests, domain, level),
                         "A JSON object describing the candidate.", llm, _build_profile, res.warnings)
        res.profile = prof or CandidateProfile(
            keywords=[k.strip() for k in re.split(r"[,;]", interests) if k.strip()][:12],
            research_areas=[k.strip() for k in re.split(r"[,;]", interests) if k.strip()][:5],
            summary="Profile could not be parsed; using your interests only.")
        emit("Agent 1: Profile Analyzer", "done", f"{len(res.profile.keywords)} keywords found.")
        _sleep(cooldown, emit, "Agent 1: Profile Analyzer")

        # ---- Agent 2: search
        stage = "search"
        emit("Agent 2: Web Search Specialist", "start", "Planning queries...")
        qp = (f"Write {max_queries} web search queries to find OFFICIAL, currently open, fully funded {level} "
              f"scholarships in: {', '.join(countries) or 'any country'}. Candidate keywords: "
              f"{', '.join(res.profile.keywords[:10])}. Year: {today.year}-{today.year + 1}. "
              f"Prefer government, ministry, university and foundation pages; include the word 'deadline'. "
              f'Return ONLY JSON: {{"queries": [str]}}')
        queries = safe_json("search", qp, "JSON with a queries list.", llm,
                            lambda d: [str(q) for q in d["queries"]][:max_queries], res.warnings)
        queries = queries or _fallback_queries(countries, level, today.year, domain)
        hits, seen = [], set()

        def _collect(qs):
            for q in qs:
                emit("Agent 2: Web Search Specialist", "info", f"Searching: {q}")
                got = ddg_search(q, per_query)
                kept = 0
                for h in got:
                    if h["url"] in seen or is_aggregator(h["url"]):
                        continue
                    seen.add(h["url"])
                    h["query"] = q
                    hits.append(h)
                    kept += 1
                res.diagnostics.append(f"Query '{q[:70]}': {len(got)} results, {kept} kept.")

        plan = list(dict.fromkeys(_site_queries(countries, level, today.year, domain)[:max_queries]
                                  + queries[:max_queries]))
        _collect(plan)
        n_first = len(hits)
        if len(hits) < 3:
            _collect(_fallback_queries(countries, level, today.year, domain)[:max_queries])
        hits.sort(key=lambda h: not is_trusted(h["url"]))      # official domains first
        hits = hits[:12]
        pages = fetch_many([h["url"] for h in hits])
        raw = [RawResult(title=h["title"], url=h["url"], snippet=h["snippet"], excerpt=pages[h["url"]][1],
                         page_ok=pages[h["url"]][0], query=h["query"]) for h in hits]
        res.diagnostics.append(f"Search: {n_first} usable hits from planned queries, {len(hits)} after fallback "
                               f"queries; {sum(r.page_ok for r in raw)} pages opened; "
                               f"{sum(is_trusted(r.url) for r in raw)} on official/academic domains.")
        if not hits:
            res.warnings.append("The search engine returned nothing (it may be throttling this server). "
                                "Retry in a minute, or rely on the curated programmes below.")
        emit("Agent 2: Web Search Specialist", "done", f"{len(raw)} pages collected.")
        _sleep(cooldown, emit, "Agent 2: Web Search Specialist")

        # ---- Agent 3a: raw -> records
        stage = "database"
        emit("Agent 3: Database & Skill Gap Mentor", "start", "Structuring search results...")
        cand = sorted([(i, r) for i, r in enumerate(raw) if r.page_ok or is_trusted(r.url)],
                      key=lambda t: not is_trusted(t[1].url))[:8]
        records, verdicts = [], {}
        if cand:
            evidence = "\n".join(f"[{i}] TITLE: {r.title}\nURL: {r.url}\nSNIPPET: {r.snippet}\n"
                                 f"PAGE: {(r.excerpt or r.snippet)[:900]}\n" for i, r in cand)
            dp = (f"From the evidence below extract genuine scholarship programmes. Skip news, listicles and "
                  f"aggregators. Use ONLY facts in the evidence; use \"\" if unknown.\n{evidence}\n"
                  f'Return ONLY JSON: {{"records": [{{"idx": int, "name": str, "provider": str, "country": str, '
                  f'"level": "MS|PhD|Postdoc|MS/PhD|Any", "funding": str, "eligibility": str, '
                  f'"requirements": "comma-separated from IELTS, TOEFL, GRE, SOP, Proposal, CV, Referees, Transcript, '
                  f'Publications, only those stated in the evidence"}}]}}')
            extracted = safe_json("database", dp, "JSON with a records list.", llm,
                                  lambda d: d["records"], res.warnings) or []
            by_idx = dict(cand)
            for e in extracted:
                try:
                    src = by_idx[int(e["idx"])]
                except Exception:
                    continue
                records.append(ScholarshipRecord(
                    name=str(e.get("name", "")).strip() or src.title, provider=str(e.get("provider", "")),
                    country=str(e.get("country", "")), level=str(e.get("level", "Any")),
                    funding=str(e.get("funding", "")), eligibility=str(e.get("eligibility", ""))[:400],
                    requirements=str(e.get("requirements", ""))[:200],
                    official_link=src.url, source="web"))
                records[-1].verification_note = str(int(e["idx"]))  # temp: evidence index
        res.diagnostics.append(f"Extraction: {len(records)} candidate records from {len(cand)} pages.")
        emit("Agent 3: Database & Skill Gap Mentor", "done", f"{len(records)} candidate records.")
        _sleep(cooldown, emit, "Agent 3: Database & Skill Gap Mentor")

        # ---- Agent 4: verification gatekeeper
        stage = "verify"
        emit("Agent 4: Verification Officer", "start", "Checking links, domains, deadlines...")
        if records:
            ev = "\n".join(f"[{r.verification_note}] {r.name} | {r.official_link}\nPAGE: "
                           f"{(raw[int(r.verification_note)].excerpt or raw[int(r.verification_note)].snippet)[:700]}\n" for r in records)
            vp = (f"For each item decide: is it a genuine scholarship programme (not a news article, blog or "
                  f"listicle), and is the page an official source of the provider? Reject when unsure.\n{ev}\n"
                  f'Return ONLY JSON: {{"checks": [{{"idx": int, "genuine": bool, "official": bool, "reason": str}}]}}')
            checks = safe_json("verify", vp, "JSON with a checks list.", llm,
                               lambda d: d["checks"], res.warnings) or []
            for c in checks:
                try:
                    verdicts[int(c["idx"])] = c
                except Exception:
                    pass
        final = []
        for r in records:
            idx = int(r.verification_note)
            page = raw[idx]
            v = verdicts.get(idx)
            text = page.excerpt or page.snippet
            trusted = is_trusted(r.official_link)
            dl, status = extract_deadline(page.excerpt, today)
            reasons = []
            if v is not None:
                if not v.get("genuine"):
                    reasons.append(f"verifier: not a genuine programme page ({v.get('reason', '')})")
                elif not (v.get("official") or trusted):
                    reasons.append(f"verifier: not an official source ({v.get('reason', '')})")
            elif not (trusted and re.search(r"scholarship|fellowship|stipend|funding", text, re.I)):
                reasons.append("no verifier verdict and weak evidence")
            if not page.page_ok and not trusted:
                reasons.append("link did not open")
            if status == "expired":
                reasons.append(f"deadline already passed ({dl})")
            if reasons:
                res.rejected.append({"name": r.name, "link": r.official_link, "reason": "; ".join(reasons)})
                continue
            r.verified = True
            if dl:
                r.deadline, r.days_left = dl.isoformat(), tk.days_left(dl.isoformat(), today)
                r.confidence = "High" if trusted else "Medium"
                r.verification_note = ("Official domain, " if trusted else "Verifier confirmed official page, ") + \
                                      "link opened, deadline found on the page."
            else:
                r.confidence = "Medium"
                r.verification_note = ("Official domain" if trusted else "Verifier confirmed official page") + \
                    (", link opened." if page.page_ok else ", but the page could not be read.") + \
                    " No deadline found: check the site."
            r.fit_score = score_fit(res.profile, r)
            final.append(r)

        # Second evidence source: verify curated programme links live (official domain + link opens)
        if len(final) < 3:
            emit("Agent 4: Verification Officer", "info", "Live-checking curated programme links...")
            seeds = seed_records(countries, level, label="curated", limit=8)
            have = {x.name.lower() for x in final}
            seed_pages = fetch_many([s.official_link for s in seeds])
            added = 0
            for s in seeds:
                ok, text = seed_pages[s.official_link]
                if not (ok and is_trusted(s.official_link)) or s.name.lower() in have:
                    continue
                dl, status = extract_deadline(text, today)
                s.verified = True
                if dl and status == "future":
                    s.deadline, s.days_left = dl.isoformat(), tk.days_left(dl.isoformat(), today)
                    s.confidence = "High"
                    s.verification_note = "Curated programme; official link opened live; deadline found on the page."
                else:
                    s.confidence = "Medium"
                    s.verification_note = ("Curated programme; official link opened live. Deadline shown is the "
                                           "typical yearly date, NOT confirmed: check the site.")
                s.fit_score = score_fit(res.profile, s)
                final.append(s)
                added += 1
            res.diagnostics.append(f"Curated list: {added} of {len(seeds)} official links opened and were added.")

        res.diagnostics.append(f"Verification: {len(final)} verified, {len(res.rejected)} rejected.")
        res.records = final
        emit("Agent 4: Verification Officer", "done",
             f"{len(final)} verified, {len(res.rejected)} rejected.")

        if not final:
            res.records = seed_records(countries, level, label="curated", limit=8)
            for r in res.records:
                r.fit_score = score_fit(res.profile, r)
            res.warnings.append("Nothing could be verified live (links unreachable from this server). Showing the "
                                "curated list, NOT verified: check every deadline on the official site.")
        res.records.sort(key=lambda r: (-(r.verified), r.days_left if r.days_left is not None else 9999))
        _sleep(cooldown, emit, "Agent 4: Verification Officer")

        # ---- Agent 3b: gap report (on verified records only)
        stage = "gap"
        emit("Agent 3: Database & Skill Gap Mentor", "start", "Analysing skill gaps...")
        brief = "\n".join(f"- {r.name} ({r.country}, {r.level}): {r.eligibility[:200]}" for r in res.records[:6])
        gp = (f"Candidate: {res.profile.model_dump_json()}\nScholarships:\n{brief}\n"
              f"Give honest strengths and gaps per scholarship, using only the eligibility text where available.\n"
              f'Return ONLY JSON: {{"strengths": [str], "common_gaps": [str], "items": [{{"scholarship": str, '
              f'"weaknesses": [str], "recommendations": [str], "priority": "High|Medium|Low", "prep_time": str}}]}}')
        res.gap_report = safe_json("database", gp, "A gap report as JSON.", llm,
                                   lambda d: GapReport(**d), res.warnings)
        attach_gaps(res.records, res.gap_report)
        emit("Agent 3: Database & Skill Gap Mentor", "done", "Gap report ready.")

        # ---- Agent 5: tracker summary (optional)
        stage = "tracker"
        df = tk.build_tracker_df(res.records)
        m = tk.metrics(df)
        if not lite:
            _sleep(cooldown, emit, "Agent 5: Tracker")
            emit("Agent 5: Tracker", "start", "Writing tracker briefing...")
            try:
                res.tracker_summary = run_agent(
                    "tracker", f"In ONE short paragraph, brief the applicant. Numbers: {m}. Nearest deadlines: "
                    f"{df.sort_values('Days left').head(3)[['Scholarship','Deadline']].to_dict('records')}. "
                    f"Do not invent dates.", "One paragraph.", llm, retries=1).strip()
            except Exception:
                res.tracker_summary = ""
        emit("Agent 5: Tracker", "done", "Tracker ready.")
    except RateLimited as e:
        res.rate_limited = True
        res.failed_stage = stage
        res.error = "Groq rate limit reached. Please wait about a minute."
        emit(stage, "error", res.error)
    except Exception as e:  # noqa: BLE001
        res.failed_stage = stage
        res.error = f"{type(e).__name__}: {str(e)[:300]}"
        emit(stage, "error", res.error)
    return res
