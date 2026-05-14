# Runtime load tracer.
#
# Auto-imported by every Python process started with PYTHONPATH=.
# When LOAD_TRACE_DIR is set, registers an atexit hook that walks
# sys.modules at shutdown and writes the file paths (filtered to those
# under LOAD_TRACE_ROOT) into {LOAD_TRACE_DIR}/{LOAD_TRACE_TAG}_pid{PID}.txt.
#
# Used to inventory which released files are actually touched by each
# end-to-end smoke. Union the per-experiment traces, diff against the full
# .py list, and the residual is dead code (relative to that smoke set).
import atexit
import os
import sys

_DIR = os.environ.get("LOAD_TRACE_DIR", "")
if _DIR:
    _TAG = os.environ.get("LOAD_TRACE_TAG", "default")
    _ROOT = os.path.realpath(os.environ.get("LOAD_TRACE_ROOT", os.getcwd()))

    os.makedirs(_DIR, exist_ok=True)

    def _dump():
        seen = set()
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            try:
                rp = os.path.realpath(f)
            except OSError:
                continue
            if rp.startswith(_ROOT):
                seen.add(rp)
        path = os.path.join(_DIR, f"{_TAG}_pid{os.getpid()}.txt")
        try:
            with open(path, "w") as h:
                for p in sorted(seen):
                    h.write(p + "\n")
        except OSError:
            pass

    atexit.register(_dump)
