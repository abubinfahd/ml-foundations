"""Verify that the ml-foundations environment is set up correctly.

Run after installing requirements.txt:

    python scripts/verify_setup.py
"""

import ctypes
import importlib
import platform
import sys

MIN_PYTHON = (3, 10)
MIN_RAM_GB = 7.0  # a nominal 8 GB machine reports slightly less

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "torch"]


def check_python() -> bool:
    ok = sys.version_info >= MIN_PYTHON
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] Python {platform.python_version()} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    return ok


def _total_ram_gb() -> float | None:
    """Total physical RAM in GB, or None if it can't be determined."""
    # Windows: ask the kernel via GlobalMemoryStatusEx.
    if sys.platform.startswith("win"):
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / 1024**3
        return None
    # Linux / WSL: read MemTotal from /proc/meminfo.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def check_ram() -> bool:
    total_gb = _total_ram_gb()
    if total_gb is None:
        print("[WARN] Could not determine total RAM — skipping RAM check")
        return True
    ok = total_gb >= MIN_RAM_GB
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] RAM {total_gb:.1f} GB (need >= 8 GB)")
    return ok


def check_packages() -> bool:
    all_ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"[OK  ] {name} {version}")
        except ImportError:
            print(f"[FAIL] {name} is not installed — did you run: pip install -r requirements.txt ?")
            all_ok = False
    return all_ok


def main() -> None:
    results = [check_python(), check_ram(), check_packages()]
    print()
    if all(results):
        print("Setup OK")
    else:
        print("Setup INCOMPLETE — fix the [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
