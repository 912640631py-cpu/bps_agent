"""Resolve BPS report sections from the current BPS Run's table of contents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SECTION_ID = re.compile(r"\d+(?:\.\d+)*")

_REQUIRED_SECTION_PATHS = (
    ("Synopsis", "Test parameters"),
    ("Synopsis", "Test Criteria"),
    ("Synopsis", "Summary of Results"),
    ("Test Environment", "Interfaces"),
)

_PERFORMANCE_TIMESERIES_PATHS = (
    ("Aggregate Stats", "Detail", "Ethernet Data Rates"),
    ("Aggregate Stats", "Detail", "Concurrent Flows"),
    ("Aggregate Stats", "Detail", "Flow Rates"),
)

_COMPONENT_SUMMARY_TITLES = frozenset(
    {
        "Component Results",
        "Application Transactions Summary",
        "Application Summary",
        "IP Summary",
        "Frame Data Rate Summary",
        "TCP Summary",
        "UDP Summary",
        "RTP Summary",
        "Frame Latency Summary",
        "Component Flow Counts",
        "Component Summary",
    }
)

_AGGREGATE_SUMMARY_TITLES = frozenset(
    {
        "Ethernet Summary",
        "ARP Summary",
        "Router Summary",
    }
)

_UNSAFE_EXPORT_PATHS = frozenset(
    {
        ("Aggregate Stats", "Detail", "Ethernet Errors"),
    }
)


@dataclass(frozen=True)
class ReportSection:
    section_id: str
    title: str
    parent_id: str | None
    path: tuple[str, ...]


@dataclass(frozen=True)
class ReportSectionSelection:
    sections: tuple[ReportSection, ...]
    required_missing: tuple[str, ...]
    ambiguous_required: tuple[dict[str, Any], ...]
    optional_missing_by_component: dict[str, tuple[str, ...]]
    unsafe_sections_skipped: tuple[dict[str, Any], ...]

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(section.section_id for section in self.sections)

    def as_dict(
        self,
        *,
        toc_section_count: int,
        mode: str = "minimal-analysis-by-title-and-parent-path",
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "toc_section_count": toc_section_count,
            "selected_section_count": len(self.sections),
            "selected_sections": [
                {
                    "id": section.section_id,
                    "title": section.title,
                    "parent_id": section.parent_id,
                    "parent_path": list(section.path[:-1]),
                    "path": list(section.path),
                }
                for section in self.sections
            ],
            "required_missing": list(self.required_missing),
            "ambiguous_required": list(self.ambiguous_required),
            "optional_missing_by_component": {
                title: list(missing)
                for title, missing in self.optional_missing_by_component.items()
            },
            "unsafe_sections_skipped": list(self.unsafe_sections_skipped),
        }


def _normalized_section_id(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = str(value)
    elif isinstance(value, str):
        candidate = value.strip()
    else:
        return None
    return candidate if _SECTION_ID.fullmatch(candidate) else None


def _normalized_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.casefold() if character.isalnum())


def _section_pair(document: dict[str, Any]) -> tuple[str, str] | None:
    id_keys = frozenset({"sectionid", "sectionnumber", "number", "id"})
    title_keys = frozenset(
        {"sectionname", "name", "title", "label", "text", "displayname", "caption"}
    )
    section_id = next(
        (
            normalized
            for key, value in document.items()
            if _normalized_key(key) in id_keys
            if (normalized := _normalized_section_id(value)) is not None
        ),
        None,
    )
    if section_id is None:
        return None
    for key, value in document.items():
        if _normalized_key(key) not in title_keys:
            continue
        if not isinstance(value, str):
            continue
        title = value.strip()
        if title and title != section_id and not _SECTION_ID.fullmatch(title) and len(title) <= 500:
            return section_id, title
    return None


def extract_report_sections(payload: Any) -> tuple[ReportSection, ...]:
    """Extract titled sections from the known BPS TOC response shapes."""

    found: dict[str, str] = {}

    def remember(section_id: str, title: str) -> None:
        clean_title = title.strip()
        previous = found.get(section_id)
        if clean_title and (previous is None or len(clean_title) > len(previous)):
            found[section_id] = clean_title

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            pair = _section_pair(value)
            if pair is not None:
                remember(*pair)
            for key, child in value.items():
                key_id = _normalized_section_id(key)
                if key_id is not None and isinstance(child, str):
                    title = child.strip()
                    if title and not _SECTION_ID.fullmatch(title):
                        remember(key_id, title)
                walk(child)
            return
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                section_id = _normalized_section_id(value[0])
                title = value[1]
                if section_id is not None and isinstance(title, str):
                    remember(section_id, title)
            for child in value:
                walk(child)
            return
        if isinstance(value, str):
            for line in value.splitlines():
                match = re.match(
                    r'^\s*(\d+(?:\.\d+)*)\s*[,;|\t]\s*"?([^",;|\t]+)',
                    line,
                )
                if match:
                    remember(match.group(1), match.group(2))

    walk(payload)

    def parent_id(section_id: str) -> str | None:
        candidate = section_id.rsplit(".", 1)[0] if "." in section_id else None
        return candidate if candidate in found else None

    def section_path(section_id: str) -> tuple[str, ...]:
        parts = section_id.split(".")
        return tuple(
            found[prefix]
            for end in range(1, len(parts) + 1)
            if (prefix := ".".join(parts[:end])) in found
        )

    return tuple(
        ReportSection(
            section_id=section_id,
            title=title,
            parent_id=parent_id(section_id),
            path=section_path(section_id),
        )
        for section_id, title in sorted(
            found.items(), key=lambda item: tuple(int(part) for part in item[0].split("."))
        )
    )


def _path_ends_with(path: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    if len(path) < len(expected):
        return False
    return tuple(part.casefold() for part in path[-len(expected) :]) == tuple(
        part.casefold() for part in expected
    )


def resolve_minimal_analysis_sections(
    sections: tuple[ReportSection, ...],
) -> ReportSectionSelection:
    """Find export Section IDs by section title and semantic parent path."""

    selected: dict[str, ReportSection] = {}
    required_missing: list[str] = []
    ambiguous_required: list[dict[str, Any]] = []

    for path in _REQUIRED_SECTION_PATHS:
        matches = [section for section in sections if _path_ends_with(section.path, path)]
        label = " > ".join(path)
        if not matches:
            required_missing.append(label)
            continue
        if len(matches) > 1:
            ambiguous_required.append(
                {"path": list(path), "matches": [section.section_id for section in matches]}
            )
        for section in matches:
            selected[section.section_id] = section

    component_roots = [
        section for section in sections if section.title.casefold().startswith("test results for ")
    ]
    optional_missing: dict[str, tuple[str, ...]] = {}
    component_selected_count = 0
    component_titles = {title.casefold(): title for title in _COMPONENT_SUMMARY_TITLES}
    for root in component_roots:
        children = [section for section in sections if section.parent_id == root.section_id]
        present = {child.title.casefold() for child in children}
        matched = [child for child in children if child.title.casefold() in component_titles]
        for section in matched:
            selected[section.section_id] = section
        component_selected_count += len(matched)
        optional_missing[root.title] = tuple(
            sorted(title for folded, title in component_titles.items() if folded not in present)
        )

    if not component_roots:
        required_missing.append("Test Results for * component root")
    elif component_selected_count == 0:
        required_missing.append("Test Results for * summary sections")

    aggregate_titles = {title.casefold() for title in _AGGREGATE_SUMMARY_TITLES}
    for root in sections:
        if root.title.casefold() != "aggregate stats":
            continue
        for section in sections:
            if (
                section.parent_id == root.section_id
                and section.title.casefold() in aggregate_titles
            ):
                selected[section.section_id] = section

    unsafe_skipped: list[dict[str, Any]] = []
    for section_id, section in tuple(selected.items()):
        if any(_path_ends_with(section.path, path) for path in _UNSAFE_EXPORT_PATHS):
            unsafe_skipped.append({"id": section.section_id, "path": list(section.path)})
            del selected[section_id]

    resolved = tuple(
        sorted(
            selected.values(),
            key=lambda section: tuple(int(part) for part in section.section_id.split(".")),
        )
    )
    return ReportSectionSelection(
        sections=resolved,
        required_missing=tuple(required_missing),
        ambiguous_required=tuple(ambiguous_required),
        optional_missing_by_component=optional_missing,
        unsafe_sections_skipped=tuple(unsafe_skipped),
    )


def resolve_performance_timeseries_sections(
    sections: tuple[ReportSection, ...],
) -> ReportSectionSelection:
    """Resolve the timestamped aggregate performance tables for a separate CSV."""

    selected: dict[str, ReportSection] = {}
    required_missing: list[str] = []
    ambiguous_required: list[dict[str, Any]] = []
    for path in _PERFORMANCE_TIMESERIES_PATHS:
        matches = [section for section in sections if _path_ends_with(section.path, path)]
        if not matches:
            required_missing.append(" > ".join(path))
            continue
        if len(matches) > 1:
            ambiguous_required.append(
                {"path": list(path), "matches": [section.section_id for section in matches]}
            )
        for section in matches:
            selected[section.section_id] = section

    resolved = tuple(
        sorted(
            selected.values(),
            key=lambda section: tuple(int(part) for part in section.section_id.split(".")),
        )
    )
    return ReportSectionSelection(
        sections=resolved,
        required_missing=tuple(required_missing),
        ambiguous_required=tuple(ambiguous_required),
        optional_missing_by_component={},
        unsafe_sections_skipped=(),
    )
