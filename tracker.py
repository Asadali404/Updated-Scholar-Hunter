"""Agent 4 logic: deadlines, urgency and application tracking (pure Python)."""
from datetime import date, datetime
from typing import Optional
import pandas as pd

STATUSES = ["Not started", "In progress", "Applied"]


def days_left(deadline_iso: str, today: Optional[date] = None) -> Optional[int]:
    if not deadline_iso:
        return None
    today = today or date.today()
    try:
        return (datetime.strptime(deadline_iso, "%Y-%m-%d").date() - today).days
    except ValueError:
        return None


def urgency(days: Optional[int]) -> str:
    if days is None:
        return "⚪ Deadline unknown"
    if days < 0:
        return "⛔ Expired"
    if days <= 14:
        return f"🔴 Critical ({days}d)"
    if days <= 30:
        return f"🟠 Soon ({days}d)"
    if days <= 60:
        return f"🟡 Upcoming ({days}d)"
    return f"🟢 Comfortable ({days}d)"


def build_tracker_df(records) -> pd.DataFrame:
    rows = [{
        "Scholarship": r.name, "Country": r.country, "Deadline": r.deadline or "Not published",
        "Days left": r.days_left, "Urgency": urgency(r.days_left), "Status": "Not started",
    } for r in records]
    cols = ["Scholarship", "Country", "Deadline", "Days left", "Urgency", "Status"]
    return pd.DataFrame(rows, columns=cols)


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Total": 0, "Applied": 0, "Pending": 0, "Remaining": 0, "Critical": 0}
    applied = int((df["Status"] == "Applied").sum())
    pending = int((df["Status"] == "In progress").sum())
    remaining = int((df["Status"] == "Not started").sum())
    crit = df[(df["Status"] != "Applied") & df["Days left"].notna() & (df["Days left"] <= 14) & (df["Days left"] >= 0)]
    return {"Total": len(df), "Applied": applied, "Pending": pending,
            "Remaining": remaining, "Critical": len(crit)}


def critical_list(df: pd.DataFrame, within_days: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[(df["Status"] != "Applied") & df["Days left"].notna()
             & (df["Days left"] >= 0) & (df["Days left"] <= within_days)]
    return sub.sort_values("Days left")[["Scholarship", "Deadline", "Days left", "Urgency", "Status"]]
