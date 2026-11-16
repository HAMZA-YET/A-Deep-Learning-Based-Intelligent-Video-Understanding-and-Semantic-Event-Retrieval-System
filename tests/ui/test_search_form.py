"""Submission lifecycle regressions; browser focus is checked separately."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from scenemind_mvp.ui import state

ROOT = Path(__file__).resolve().parents[2]
STUDENTS = "c242ea439bad72789b87929c627e1ab64a67a9c211bdb82decd79dbd037b3ce0"


def test_search_form_preserves_draft_and_only_submits_complete_query():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    app.selectbox(key="sm_video").set_value(STUDENTS).run()
    assert not app.exception
    field = app.text_input(key="sm_query")
    assert field.proto.form_id == "sm_search_form"
    assert not field.proto.HasField("live_debounce_ms")
    initial = app.session_state[state.KEY]["id"]
    query = "a person walking in green shirt"
    field.set_value(query)
    assert app.session_state[state.KEY]["id"] == initial
    app.button(key="sm_search_submit").click().run()
    assert not app.exception
    first = app.session_state[state.KEY]
    assert first["query"] == query and first["status"] == "done"
    assert app.text_input(key="sm_query").value == query
    assert len(app.get("video")) == 1
    # A playback-only rerun must not submit a draft or discard it.
    app.text_input(key="sm_query").set_value("a person walking")
    app.segmented_control(key="sm_playback_mode").set_value("Original Video").run()
    assert app.session_state[state.KEY]["id"] == first["id"]
    # AppTest omits unsubmitted form values on outside-widget reruns.
    # Browser-owned draft retention is verified in the actual browser.
    app.text_input(key="sm_query").set_value("a person walking")
    app.button(key="sm_search_submit").click().run()
    assert not app.exception
    assert app.session_state[state.KEY]["id"] != first["id"]
    assert app.session_state[state.KEY]["query"] == "a person walking"
    app.text_input(key="sm_query").set_value("")
    app.button(key="sm_search_submit").click().run()
    assert app.session_state[state.KEY]["result"] is None
    assert app.segmented_control(key="sm_unified_view").value == "Show All"
    # Clear must also work on a never-submitted draft from idle state.
    app.text_input(key="sm_query").set_value("unfinished draft")
    app.button(key="sm_clear_search").click().run()
    assert not app.exception
    assert app.text_input(key="sm_query").value == ""
    assert app.session_state[state.KEY]["status"] == "idle"
    assert len(app.get("video")) == 1
