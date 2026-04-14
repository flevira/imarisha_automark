from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.qr import parse_qr_data


def test_parse_qr_data_includes_student_name_field() -> None:
    parsed = parse_qr_data(
        "type=EXAM;studentId=12345;studentName=Jane Doe;examId=ENG-01"
    )

    assert parsed["studentid"] == "12345"
    assert parsed["studentname"] == "Jane Doe"
    assert parsed["examid"] == "ENG-01"
