"""Excel export (openpyxl via pandas)."""
import io
import pandas as pd
from tracker import urgency


def records_df(records) -> pd.DataFrame:
    return pd.DataFrame([{
        "Scholarship": r.name, "Provider": r.provider, "Country": r.country, "Level": r.level,
        "Funding": r.funding, "Deadline": r.deadline or "Not published", "Days left": r.days_left,
        "Urgency": urgency(r.days_left), "Fit score": r.fit_score, "Confidence": r.confidence,
        "Verified": "Yes" if r.verified else "No", "Verification note": r.verification_note,
        "Official link": r.official_link, "Eligibility": r.eligibility,
    } for r in records])


def to_excel(result, tracker_df=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        p = result.profile
        if p:
            pd.DataFrame({"Field": ["Degree", "CGPA", "Publications", "Skills", "Research areas", "Keywords", "Summary"],
                          "Value": [p.degree, p.cgpa, p.publications, ", ".join(p.skills),
                                    ", ".join(p.research_areas), ", ".join(p.keywords), p.summary]}
                         ).to_excel(xw, sheet_name="Profile", index=False)
        records_df(result.records).to_excel(xw, sheet_name="Opportunities", index=False)
        if result.gap_report:
            pd.DataFrame([{"Scholarship": i.scholarship, "Priority": i.priority, "Prep time": i.prep_time,
                           "Weaknesses": "; ".join(i.weaknesses), "Recommendations": "; ".join(i.recommendations)}
                          for i in result.gap_report.items]).to_excel(xw, sheet_name="Gap Analysis", index=False)
        if tracker_df is not None:
            tracker_df.to_excel(xw, sheet_name="Tracker", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) if c.value else 0) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 60)
    return buf.getvalue()
