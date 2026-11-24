"""Chặn hồi quy trong `requirements.txt`: không khai báo trùng, các gói quyết định kết quả phải pin."""
import re
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"

#: Phiên bản đã kiểm trên Python 3.12 (xem docs/DATA.md §3). Đổi pin thì phải đổi cả test này.
PINNED = {"pandas": "3.1.1", "pyarrow": "26.0.1", "scikit-learn": "1.9.1"}
#: Gói bắt buộc phải có mặt (joblib được baseline.py import trực tiếp).
REQUIRED = {"joblib", *PINNED}


def parsed() -> dict[str, str | None]:
    """{tên gói: phiên bản pin} cho từng dòng khai báo; `None` = không pin."""
    entries: dict[str, str | None] = {}
    duplicates = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        name = name.strip().lower().replace("_", "-")
        if name in entries:
            duplicates.append(name)
        entries[name] = version.strip() if sep else None
    assert not duplicates, f"khai báo trùng: {duplicates}"
    return entries


def test_no_duplicate_declarations():
    entries = parsed()
    names = [line.split("==")[0].strip().lower().replace("_", "-")
             for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    assert len(names) == len(set(names)), f"trùng: {sorted({n for n in names if names.count(n) > 1})}"
    assert entries  # file không rỗng


def test_data_critical_packages_are_pinned_to_verified_versions():
    entries = parsed()
    for name, version in PINNED.items():
        assert entries.get(name) == version, f"{name} phải pin =={version}, đang là {entries.get(name)!r}"


def test_joblib_is_declared():
    assert "joblib" in parsed()


def test_no_unpinned_duplicate_of_pinned_packages():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    for name in PINNED:
        assert not re.search(rf"^{re.escape(name)}\s*$", text, flags=re.MULTILINE), f"{name} còn dòng không pin"
