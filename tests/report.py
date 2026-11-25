"""Helpers for parsing Ledger markdown reports in tests."""

import re


def section(text, heading):
    """Return the body of a level-2 markdown section, excluding its heading."""
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert m, f"missing section: {heading}"
    return m.group(1)


def table_rows(section_text):
    """Parse markdown table data rows into lists of stripped cell strings."""
    rows = []
    for line in section_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows[1:] if rows else []
