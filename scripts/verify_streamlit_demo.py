"""Actual UI regression for the unified dashboard; historical verifier is archived."""
from pathlib import Path
import subprocess,sys,json,time
root=Path(__file__).resolve().parents[1]
target='tests/ui/test_main_app.py::test_existing_demo_search_and_timestamp_navigation'
started=time.perf_counter()
result=subprocess.run([sys.executable,'-m','pytest',target,'-q','--tb=short'],cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
report={'status':'PASS' if result.returncode==0 else 'FAIL','target':target,'elapsed_seconds':time.perf_counter()-started,'stdout':result.stdout,'stderr':result.stderr,'implementation':'Unified app.py dashboard; original verifier retained under docs/ui-redesign/legacy-checks'}
(root/'docs/ui-redesign'/'verify_streamlit_demo.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(result.stdout);print(result.stderr,file=sys.stderr)
raise SystemExit(result.returncode)
