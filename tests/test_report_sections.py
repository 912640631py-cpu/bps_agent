from __future__ import annotations

from bps_agent.report_sections import (
    extract_report_sections,
    resolve_minimal_analysis_sections,
)


def test_resolves_current_run_ids_by_title_and_parent_path() -> None:
    toc = {
        "tableOfContents": [
            {"Section ID": "10", "Section Name": "Report"},
            {"section_id": "10.2", "title": "Synopsis"},
            {"sectionId": "10.2.7", "sectionName": "Test parameters"},
            {"sectionId": "10.2.8", "sectionName": "Test Criteria"},
            {"sectionId": "10.2.9", "sectionName": "Summary of Results"},
            {"sectionId": "10.3", "sectionName": "Appendix"},
            {"sectionId": "10.3.7", "sectionName": "Test parameters"},
            {"sectionId": "12", "sectionName": "Test Environment"},
            {"sectionId": "12.12", "sectionName": "Interfaces"},
            {"sectionId": "17", "sectionName": "Test Results for AppSim"},
            {"sectionId": "17.6", "sectionName": "Component Results"},
            {"sectionId": "17.11", "sectionName": "Application Summary"},
            {"sectionId": "19", "sectionName": "Aggregate Stats"},
            {"sectionId": "19.4", "sectionName": "Ethernet Summary"},
        ]
    }

    sections = extract_report_sections(toc)
    selection = resolve_minimal_analysis_sections(sections)

    assert selection.required_missing == ()
    assert selection.section_ids == (
        "10.2.7",
        "10.2.8",
        "10.2.9",
        "12.12",
        "17.6",
        "17.11",
        "19.4",
    )
    assert "10.3.7" not in selection.section_ids
    selected = selection.as_dict(toc_section_count=len(sections))["selected_sections"]
    assert selected[0]["parent_path"] == ["Report", "Synopsis"]


def test_extracts_mapping_pair_and_text_toc_shapes() -> None:
    sections = extract_report_sections(
        {
            "mapping": {"1": "Synopsis"},
            "pairs": [["1.2", "Test parameters"]],
            "text": "2,Test Environment\n2.3|Interfaces",
        }
    )

    assert [(section.section_id, section.title, section.path) for section in sections] == [
        ("1", "Synopsis", ("Synopsis",)),
        ("1.2", "Test parameters", ("Synopsis", "Test parameters")),
        ("2", "Test Environment", ("Test Environment",)),
        ("2.3", "Interfaces", ("Test Environment", "Interfaces")),
    ]
