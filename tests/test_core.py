import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from datetime import date
from tools import extract_deadline, extract_json, is_trusted
import tracker as tk
from crew import demo_result, next_occurrence


def test_deadline_future():
    d, s = extract_deadline("Application deadline: 15 November 2026.", date(2026, 10, 4))
    assert s == "future" and d == date(2026, 11, 15)


def test_deadline_expired():
    d, s = extract_deadline("Deadline was March 1, 2025", date(2026, 10, 4))
    assert s == "expired"


def test_trust():
    assert is_trusted("https://www.daad.de/en/")
    assert is_trusted("https://someuni.edu.pk/x")
    assert not is_trusted("https://random-scholarship-blog.com/post")


def test_json_tolerant():
    assert extract_json('noise ```json {"a": 1} ``` more') == {"a": 1}


def test_next_occurrence_never_past():
    assert next_occurrence("01-15", date(2026, 10, 4)) == "2027-01-15"


def test_demo_and_tracker():
    r = demo_result(["Germany", "UK"], "MS")
    assert r.records and all(x.days_left >= 0 for x in r.records)
    df = tk.build_tracker_df(r.records)
    assert tk.metrics(df)["Total"] == len(r.records)
