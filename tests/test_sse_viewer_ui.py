from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_PATH = Path(__file__).resolve().parent / "sse_viewer.html"
README_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.en.md",
    REPO_ROOT / "README.ko.md",
)
BENCHMARK_REPORT_PATH = REPO_ROOT / "evals" / "benchmark_report.md"
README_ASSET_RE = re.compile(r"!\[[^\]]*\]\((docs/assets/[^)\s]+)\)")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


@dataclass
class _HtmlNode:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[_HtmlNode] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    parent: _HtmlNode | None = None

    @property
    def id(self) -> str:
        return self.attrs.get("id", "")

    @property
    def classes(self) -> set[str]:
        raw = self.attrs.get("class", "")
        return {token for token in raw.split() if token}

    @property
    def text(self) -> str:
        own = "".join(self.text_parts)
        nested = "".join(child.text for child in self.children)
        return f"{own}{nested}"

    def iter_descendants(self) -> list[_HtmlNode]:
        result: list[_HtmlNode] = []
        for child in self.children:
            result.append(child)
            result.extend(child.iter_descendants())
        return result

    def find_by_id(self, elem_id: str) -> _HtmlNode | None:
        for node in self.iter_descendants():
            if node.id == elem_id:
                return node
        return None

    def find_all(
        self,
        *,
        tag: str | None = None,
        class_name: str | None = None,
    ) -> list[_HtmlNode]:
        matches: list[_HtmlNode] = []
        for node in self.iter_descendants():
            if tag is not None and node.tag != tag:
                continue
            if class_name is not None and class_name not in node.classes:
                continue
            matches.append(node)
        return matches


class _DomBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode(tag="document")
        self._stack: list[_HtmlNode] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v if v is not None else "") for k, v in attrs}
        node = _HtmlNode(tag=tag.lower(), attrs=attr_map, parent=self._stack[-1])
        self._stack[-1].children.append(node)
        if node.tag not in VOID_ELEMENTS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v if v is not None else "") for k, v in attrs}
        node = _HtmlNode(tag=tag.lower(), attrs=attr_map, parent=self._stack[-1])
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        target = tag.lower()
        for idx in range(len(self._stack) - 1, 0, -1):
            if self._stack[idx].tag == target:
                del self._stack[idx:]
                break

    def handle_data(self, data: str) -> None:
        self._stack[-1].text_parts.append(data)


def _load_viewer_html() -> str:
    assert VIEWER_PATH.is_file(), f"Missing viewer file: {VIEWER_PATH}"
    return VIEWER_PATH.read_text(encoding="utf-8")


def _parse_dom(html_text: str) -> _HtmlNode:
    builder = _DomBuilder()
    builder.feed(html_text)
    builder.close()
    return builder.root


def _extract_script_body(dom: _HtmlNode) -> str:
    scripts = dom.find_all(tag="script")
    assert scripts, "Expected inline <script> in sse_viewer.html"
    return "\n".join(node.text for node in scripts)


def _extract_style_body(dom: _HtmlNode) -> str:
    styles = dom.find_all(tag="style")
    assert styles, "Expected inline <style> in sse_viewer.html"
    return "\n".join(node.text for node in styles)


def _extract_i18n_dict(script_body: str) -> dict[str, Any]:
    marker_match = re.search(
        r"/\*\s*__I18N_DICT_START__\s*\*/\s*const\s+I18N_DICT\s*=\s*(\{.*?\})\s*;\s*/\*\s*__I18N_DICT_END__\s*\*/",
        script_body,
        flags=re.DOTALL,
    )
    assert marker_match is not None, (
        "I18N_DICT must be wrapped with __I18N_DICT_START__ / __I18N_DICT_END__ markers "
        "in strict JSON-compatible syntax for deterministic parsing."
    )
    parsed = json.loads(marker_match.group(1))
    assert isinstance(parsed, dict)
    return parsed


def _iter_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(value, str):
        items.append((path, value))
    elif isinstance(value, list):
        for idx, elem in enumerate(value):
            items.extend(_iter_strings(elem, f"{path}[{idx}]"))
    elif isinstance(value, dict):
        for k, v in value.items():
            sub_path = f"{path}.{k}" if path else str(k)
            items.extend(_iter_strings(v, sub_path))
    return items


def test_prompt_dock_single_primary_action_button_and_abort_support() -> None:
    html_text = _load_viewer_html()
    dom = _parse_dom(html_text)
    script_body = _extract_script_body(dom)

    assert dom.find_by_id("compareAllBtn") is None, "Duplicate compareAllBtn must be removed"

    prompt_docks = dom.find_all(class_name="prompt-dock")
    assert len(prompt_docks) == 1
    dock = prompt_docks[0]

    dock_actions = dock.find_all(class_name="dock-actions")
    assert len(dock_actions) == 1
    action_buttons = dock_actions[0].find_all(tag="button")
    assert [btn.id for btn in action_buttons] == ["sendBtn"]

    pipe_select = dock.find_by_id("playgroundPipeSelect")
    assert pipe_select is not None, "Playground mode needs inline #playgroundPipeSelect"
    option_values = [opt.attrs.get("value") for opt in pipe_select.find_all(tag="option")]
    assert set(option_values) == {"v1", "v2", "v3"}
    v3_opts = [opt for opt in pipe_select.find_all(tag="option") if opt.attrs.get("value") == "v3"]
    assert v3_opts and "selected" in v3_opts[0].attrs

    assert "new AbortController()" in script_body
    assert 'err.name === "AbortError"' in script_body or "isAbortError(" in script_body


def test_sidebar_slimmed_and_drawers_for_ingest_and_settings() -> None:
    html_text = _load_viewer_html()
    dom = _parse_dom(html_text)
    style_body = _extract_style_body(dom)
    script_body = _extract_script_body(dom)

    grid_match = re.search(
        r"\.workbench-grid\s*\{[^}]*grid-template-columns:\s*(\d+)px\s+minmax\(0,\s*1fr\)",
        style_body,
        flags=re.DOTALL,
    )
    assert grid_match is not None, "Expected .workbench-grid grid-template-columns rule"
    left_width = int(grid_match.group(1))
    assert 220 <= left_width <= 240, f"Left sidebar width must be 220-240px, got {left_width}px"
    assert ".workbench-grid.left-collapsed" in style_body

    sidebars = dom.find_all(class_name="sidebar-left")
    assert len(sidebars) == 1
    left_sidebar = sidebars[0]
    for required_id in ("sessionInput", "newSessionBtn", "clearHistoryBtn", "sessionTimeline"):
        assert left_sidebar.find_by_id(required_id) is not None
    for forbidden_id in (
        "ingestPanel",
        "advancedSettingsPanel",
        "ingestSourceInput",
        "endpointInput",
    ):
        assert left_sidebar.find_by_id(forbidden_id) is None

    assert dom.find_by_id("toggleLeftSidebarBtn") is not None

    headers = dom.find_all(class_name="top-header")
    assert len(headers) == 1
    top_header = headers[0]
    chunks_btn = top_header.find_by_id("storeChunksBadge")
    assert chunks_btn is not None and chunks_btn.tag == "button"
    assert top_header.find_by_id("openIngestDrawerBtn") is None, (
        "#storeChunksBadge should be the single Ingest drawer trigger in top header"
    )
    assert top_header.find_by_id("openSettingsDrawerBtn") is not None

    assert dom.find_by_id("drawerBackdrop") is not None
    ingest_drawer = dom.find_by_id("ingestDrawer")
    settings_drawer = dom.find_by_id("settingsDrawer")
    assert ingest_drawer is not None
    assert settings_drawer is not None

    for ingest_id in (
        "ingestSourceInput",
        "ingestPipelineSelect",
        "ingestContentInput",
        "ingestBtn",
        "ingestStatus",
    ):
        assert ingest_drawer.find_by_id(ingest_id) is not None

    for settings_id in ("endpointInput", "pipelineSelect", "topKInput", "streamCheckbox"):
        assert settings_drawer.find_by_id(settings_id) is not None

    assert '"Escape"' in script_body


def test_arena_empty_state_denoised_and_inspector_skipped_semantics() -> None:
    html_text = _load_viewer_html()
    dom = _parse_dom(html_text)
    script_body = _extract_script_body(dom)

    playground_pills = dom.find_by_id("playgroundTelemetryPills")
    assert playground_pills is not None
    assert "--" not in playground_pills.text
    assert "display: none" in playground_pills.attrs.get("style", "")

    assert "buildCompactTelemetryHtml" in script_body
    assert "selectInspectedPipeline(spec.version)" in script_body
    assert 'style.display = "none"' in script_body or "hasMetrics" in script_body

    # Trace Node 5 must use data-i18n instead of hardcoded Chinese
    trace_sub5 = dom.find_by_id("traceSub5")
    assert trace_sub5 is not None
    assert trace_sub5.attrs.get("data-i18n") == "trace_n5_sub"

    # v1 trace semantics: only nodes 1, 3, 4 are skipped; node 2 remains active/done with v1 subtitle
    assert 't("trace_n2_sub_v1")' in script_body
    assert 't("trace_n5_sub")' in script_body
    assert "is_cacheable_response 防降级缓存污染" not in script_body.split("__I18N_DICT_END__")[-1]


def test_i18n_trilingual_dictionary_key_parity_and_zero_cjk_leakage() -> None:
    html_text = _load_viewer_html()
    dom = _parse_dom(html_text)
    script_body = _extract_script_body(dom)
    i18n_dict = _extract_i18n_dict(script_body)

    assert set(i18n_dict.keys()) == {"zh", "en", "ko"}

    zh_keys = set(i18n_dict["zh"].keys())
    en_keys = set(i18n_dict["en"].keys())
    ko_keys = set(i18n_dict["ko"].keys())
    assert zh_keys == en_keys, f"Mismatch zh vs en keys: {zh_keys ^ en_keys}"
    assert zh_keys == ko_keys, f"Mismatch zh vs ko keys: {zh_keys ^ ko_keys}"

    for lang in ("zh", "en", "ko"):
        pills = i18n_dict[lang].get("quick_pills")
        assert isinstance(pills, list) and len(pills) == 4
        for pill in pills:
            assert isinstance(pill.get("label"), str) and pill["label"].strip()
            assert isinstance(pill.get("prompt"), str) and pill["prompt"].strip()

    for lang in ("en", "ko"):
        for path, text_val in _iter_strings(i18n_dict[lang], lang):
            assert CJK_RE.search(text_val) is None, (
                f"Chinese character leaked in I18N_DICT[{path!r}]: {text_val!r}"
            )

    for node in dom.iter_descendants():
        key = node.attrs.get("data-i18n")
        if key:
            assert key in zh_keys, f"Unknown data-i18n key in HTML: {key}"
        placeholder_key = node.attrs.get("data-i18n-placeholder")
        if placeholder_key:
            assert placeholder_key in zh_keys, (
                f"Unknown data-i18n-placeholder key in HTML: {placeholder_key}"
            )


def test_export_controls_and_blob_download_logic() -> None:
    dom = _parse_dom(_load_viewer_html())
    script_body = _extract_script_body(dom)

    md_btn = dom.find_by_id("exportMdBtn")
    json_btn = dom.find_by_id("exportJsonBtn")
    assert md_btn is not None and md_btn.tag == "button"
    assert json_btn is not None and json_btn.tag == "button"
    assert md_btn.attrs.get("data-i18n") == "btn_export_md"
    assert json_btn.attrs.get("data-i18n") == "btn_export_json"

    for snippet in (
        "function buildExportReport(",
        "function downloadBlob(",
        "URL.createObjectURL(",
        "URL.revokeObjectURL(",
        "rag-eval-report-",
        "toISOString()",
        "syncExportButtons",
    ):
        assert snippet in script_body, f"Missing export logic: {snippet}"

    i18n_dict = _extract_i18n_dict(script_body)
    for lang in ("zh", "en", "ko"):
        for key in (
            "btn_export_md",
            "btn_export_json",
            "export_empty_hint",
            "export_success",
        ):
            assert i18n_dict[lang].get(key), f"{lang} missing i18n key {key}"


def test_readme_local_images_exist_and_are_consistent() -> None:
    per_readme_assets: dict[str, list[str]] = {}
    for readme in README_PATHS:
        assert readme.is_file(), f"Missing {readme.name}"
        assets = README_ASSET_RE.findall(readme.read_text(encoding="utf-8"))
        assert len(set(assets)) >= 2, f"{readme.name} must embed at least 2 docs/assets images"
        for rel_path in assets:
            assert (REPO_ROOT / rel_path).is_file(), f"{readme.name} broken image: {rel_path}"
        per_readme_assets[readme.name] = sorted(set(assets))

    reference = per_readme_assets["README.md"]
    for name, assets in per_readme_assets.items():
        assert assets == reference, f"{name} image set differs from README.md"


def test_readme_benchmark_source_exists_and_is_cited() -> None:
    assert BENCHMARK_REPORT_PATH.is_file()
    report_text = BENCHMARK_REPORT_PATH.read_text(encoding="utf-8")
    generated_match = re.search(r"\*\*Generated At\*\*: `([^`]+)`", report_text)
    assert generated_match is not None
    assert "offline-heuristic" in report_text

    for readme in README_PATHS:
        text = readme.read_text(encoding="utf-8")
        assert "evals/benchmark_report.md" in text
        assert generated_match.group(1) in text, f"{readme.name} does not cite the report timestamp"
        assert "offline-heuristic" in text
