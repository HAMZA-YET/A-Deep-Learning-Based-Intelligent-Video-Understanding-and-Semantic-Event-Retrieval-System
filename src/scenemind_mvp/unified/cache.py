"""Cross-process cache-key locks and non-destructive atomic directory publication."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import os,time,threading
_LOCKS={};_GUARD=threading.Lock()

@contextmanager
def cache_lock(root: Path,key: str,timeout: float=600.):
    """Both threads and separate Streamlit/process workers serialize the same key."""
    directory=root/'.locks';directory.mkdir(parents=True,exist_ok=True);path=directory/(key+'.lock')
    with _GUARD:local=_LOCKS.setdefault(str(path.resolve()),threading.RLock())
    if not local.acquire(timeout=timeout):raise TimeoutError('Timed out waiting for this render cache')
    handle=None;locked=False;started=time.monotonic()
    try:
        handle=path.open('a+b');handle.seek(0,2)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        while True:
            try:
                handle.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True;break
            except OSError:
                if time.monotonic()-started>=timeout:raise TimeoutError('Another process is still rendering this selection')
                time.sleep(.05)
        yield
    finally:
        if handle:
            if locked:
                handle.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
            handle.close()
        local.release()

def publish(staging: Path,destination: Path,validate):
    """Never replace a destination. A non-cooperating winner is reused if valid."""
    try:staging.rename(destination)
    except OSError as error:
        if destination.exists() and validate(destination):return False
        raise RuntimeError('Render cache destination exists but is incomplete or invalid; it was preserved: '+str(destination)) from error
    return True
