"""PASS/FAIL bookkeeping shared by the verification groups.

Every check prints one line   name | computed | target | tolerance | PASS/FAIL
and is recorded; summary() prints the totals and returns the exit status.
"""
import sys
import time

_results = []
_t_start = time.time()


def _fmt(v, fmt):
    if isinstance(v, (float, int)) and not isinstance(v, bool):
        try:
            return fmt % v
        except TypeError:
            return str(v)
    return str(v)


def report(name, computed, target, tol, ok, cfmt="%.6g", tfmt="%.6g"):
    _results.append((name, bool(ok)))
    print("%-58s | %22s | %22s | %10s | %s"
          % (name, _fmt(computed, cfmt), _fmt(target, tfmt), _fmt(tol, "%.3g"), "PASS" if ok else "FAIL"), flush=True)
    return bool(ok)


def close(name, computed, target, tol, cfmt="%.4f", tfmt="%.4f"):
    computed, target = float(computed), float(target)
    return report(name, computed, target, float(tol), abs(computed - target) <= tol, cfmt, tfmt)


def rel(name, computed, target, rtol, cfmt="%.4e", tfmt="%.3e"):
    computed, target = float(computed), float(target)
    return report(name, computed, target, rtol, abs(computed / target - 1.0) <= rtol, cfmt, tfmt)


def equal(name, computed, target):
    return report(name, computed, target, 0, computed == target, "%s", "%s")


def below(name, computed, bound, cfmt="%.3e"):
    return report(name, float(computed), float(bound), "<=", float(computed) <= float(bound), cfmt, cfmt)


def info(name, value, cfmt="%.4g"):
    """an informational line that is always PASS."""
    return report(name, value, "info", "-", True, cfmt, "%s")


def section(title):
    print("\n--- %s ---" % title, flush=True)


def summary():
    n_pass = sum(1 for _, ok in _results if ok)
    n_all = len(_results)
    print("\n" + "=" * 100)
    if n_pass == n_all:
        print("ALL PASS (%d/%d)   [total runtime %.1fs]" % (n_pass, n_all, time.time() - _t_start))
    else:
        print("FAILURES (%d/%d passed)   [total runtime %.1fs]" % (n_pass, n_all, time.time() - _t_start))
        for name, ok in _results:
            if not ok:
                print("  FAIL: %s" % name)
    return 0 if n_pass == n_all else 1


def results():
    return list(_results)
