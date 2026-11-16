# Search focus repair — 2026-11-16

## Cause and scope
The shared search input used `live=True` and an on-change callback outside a form, so editing emitted reruns. Its parent was an `st.empty()` placeholder recreated before the sidebar/settings and search widgets were filled. Clearing that placeholder on reruns permits browser unmounting/focus loss. This path is shared by Original and Annotated modes; video inference is not the cause. The input key itself was already stable. No periodic refresh, polling, timer, or automatic rerun loop exists in app.py or ui modules. The one remaining st.rerun runs only after a submitted search completes.

## Changes
- app.py: stable keyed search container; keyed form with clear_on_submit=False; non-live sm_query input; Search is the only form submit button (Enter supported).
- Draft sm_query remains separate from sm_execution.query (last submitted query). No draft is copied into an execution until submit_search.
- Clear is an always-enabled button outside the form, including for browser-only unsubmitted drafts. Clear or empty submission restores Show All. Deleting text alone does not submit anything.
- No focus-forcing JavaScript/CSS, detector, tracker, index, or video-render changes.
- tests/ui/test_search_form.py: submission, draft/last-execution separation, preservation after submission, empty submit, explicit Clear, single-player regression.
- tests/unified/test_unified_integration.py: empty-query clearing now explicitly submits the form, matching new interaction semantics.

## Executed verification
Actual production browser at http://127.0.0.1:8501, Mixkit student video, same browser session for five consecutive searches, no server restart:
1. Original Video: a person walking in green shirt — 31 characters, Enter.
2. Annotated / Show All: a person wearing a black shirt — 30 characters, Search.
3. Annotated: a person walking — 16 characters, Enter.
4. Original: a person wearing a white shirt — 30 characters, Search.
5. Annotated: a person walking in green shirt — 31 characters, Enter.

All 138 characters entered individually, 200–300ms spacing, without refocusing between characters. Each DOM observation confirmed focus and exact incremental input; element ID stayed :r11:. Previous results remained until explicit submission, each submitted heading used the complete query. Draft survived a playback mode change without replacing the last submitted query. Empty submission and Clear returned to Show All, including clearing an unsubmitted draft. Actual Original and Annotated videos both advanced while typing (11.12s and 9.22s observed, readyState4, pausedFalse).

Automated affected suites: **46 passed in 139.30 seconds**, tests/ui + tests/unified. Full historical suite not rerun for this UI-only change. First test attempt used a nonexistent protobuf field; corrected to HasField(live_debounce_ms). Second exposed AppTest's omission of browser-owned unsent form state during outside-widget reruns; actual browser confirmed draft retention, and test now limits that assertion to submitted state.

Raw browser trace: search-focus-browser.json. Final tests: search-form-tests-verified.log. Earlier failed test logs retained. These tests verify input lifecycle, not clothing/action accuracy.
