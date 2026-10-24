""" System Information Context service.

    Collects static hardware metrics and operating-system/runtime metrics for the
    Produces a dataclass
"""

import os
import re
import sys
import json
import time
import shutil
import logging
import platform
import subprocess
import multiprocessing
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor


# Logging setup and constants setup
UNKNOWN = "unknown"
PROBE_TIMEOUT_SECONDS = 2.0
COLLECTION_BUDGET_SECONDS = 30.0
SYSTEM_INFO_FILENAME = "system_profile.json"
logger = logging.getLogger("subconscious")


@dataclass(frozen=True)
class StaticMetrics:
  """ Host hardware metrics that rarely change.

      Byte-valued fields (``total_ram_bytes``, ``total_vram_bytes``) are stored as
      raw byte counts in string form; conversion to GB happens only at format time.
  """
  cpu_model: str = UNKNOWN
  gpu_model: str = UNKNOWN
  accelerator: str = UNKNOWN
  logical_cores: str = UNKNOWN
  physical_cores: str = UNKNOWN
  total_ram_bytes: str = UNKNOWN
  cpu_architecture: str = UNKNOWN
  total_vram_bytes: str = UNKNOWN


@dataclass(frozen=True)
class OSMetrics:
  """ Operating-system and runtime metrics """
  os_name: str = UNKNOWN
  os_version: str = UNKNOWN
  python_version: str = UNKNOWN
  machine_architecture: str = UNKNOWN


@dataclass(frozen=True)
class SystemProfile:
  """ The structured record of collected static system information """
  os: OSMetrics
  static: StaticMetrics


class SystemInformationService:
  """ Collects, persists, and formats host system capability information  """

  def __init__(self, data_dir: str) -> None:
    """Store ``data_dir``, compute the ``System_Info_File`` path
    (``<data_dir>/system_profile.json``), and initialize an empty in-memory
    profile cache."""
    self._data_dir = data_dir
    self._system_info_file = os.path.join(data_dir, SYSTEM_INFO_FILENAME)
    self._profile: Optional[SystemProfile] = None

  # ---------------------------------------------------------------------
  # Probe guarding and normalization helpers
  # ---------------------------------------------------------------------

  def _safe(
    self,
    fn: Callable[[], object],
    timeout: Optional[float] = None,
  ) -> str:
    """Run a single probe under full error and timeout guarding.

    ``fn`` is a zero-argument probe that returns the metric value. All
    exceptions are caught and any failure yields ``UNKNOWN`` (Requirement
    5.2). Probes that shell out to external tools pass a ``timeout`` so a hung
    command cannot stall collection: the probe is run on a worker thread and,
    if it does not complete within ``timeout`` seconds, ``UNKNOWN`` is
    returned. A ``None`` or blank result is also normalized to ``UNKNOWN`` so
    an undeterminable metric never leaks into the profile.
    """
    try:
      if timeout is not None:
        with ThreadPoolExecutor(max_workers=1) as executor:
          result = executor.submit(fn).result(timeout=timeout)
      else:
        result = fn()
    except Exception:
      logger.warning("System info probe failed; recording %r", UNKNOWN, exc_info=True)
      return UNKNOWN

    if result is None:
      return UNKNOWN
    text = str(result).strip()
    return text or UNKNOWN

  @staticmethod
  def _normalize_vram(raw: object) -> str:
    """Normalize a raw VRAM reading to a stored ``total_vram_bytes`` value.

    Returns ``UNKNOWN`` when the reading is missing, non-numeric (not an
    integer), or numerically zero (or negative); any strictly positive byte
    value is preserved exactly as its string form (Requirements 1.2, 1.3).
    """
    if raw is None:
      return UNKNOWN
    try:
      value = int(str(raw).strip())
    except (TypeError, ValueError):
      return UNKNOWN
    if value <= 0:
      return UNKNOWN
    return str(value)

  # ---------------------------------------------------------------------
  # Collection
  # ---------------------------------------------------------------------

  def _collect(self) -> SystemProfile:
    """Collect Static_Metrics and OS_Metrics under the overall time budget.

    Every metric is gathered through an individually guarded probe (via
    ``_safe``), so a failure in one probe records ``UNKNOWN`` for that metric
    and collection continues with the next (Requirements 1, 2, 5.1, 5.2).
    Optional ``psutil`` is used when importable and the service falls back to
    the standard library otherwise (Requirement 5.3). GPU/VRAM/accelerator
    detection is best-effort and platform-dispatched. The whole pass is bound
    by ``COLLECTION_BUDGET_SECONDS``: once the budget is exhausted, any
    remaining metrics keep their ``UNKNOWN`` defaults, yielding a valid partial
    profile (Requirement 5.5). This method never raises and always returns a
    total profile with an entry for every defined field (Requirement 5.4).
    """
    deadline = time.monotonic() + COLLECTION_BUDGET_SECONDS

    def gather(fn: Callable[[], Any], timeout: Optional[float] = None) -> str:
      # Honor the overall collection budget: once exhausted, leave the metric
      # at its UNKNOWN default rather than starting another probe.
      if time.monotonic() >= deadline:
        return UNKNOWN
      return self._safe(fn, timeout=timeout)

    # --- Cheap, in-process metrics first -----------------------------------
    cpu_model = gather(self._probe_cpu_model, timeout=PROBE_TIMEOUT_SECONDS)
    physical_cores = gather(self._probe_physical_cores)
    logical_cores = gather(self._probe_logical_cores)
    total_ram_bytes = gather(self._probe_total_ram_bytes, timeout=PROBE_TIMEOUT_SECONDS)
    cpu_architecture = gather(self._probe_cpu_architecture)

    os_version = gather(self._probe_os_version)
    python_version = gather(self._probe_python_version)
    machine_architecture = gather(self._probe_machine_architecture)
    os_name = gather(self._probe_os_name, timeout=PROBE_TIMEOUT_SECONDS)

    # --- Expensive, shell-based metrics last -------------------------------
    # GPU / VRAM / accelerator are derived from a single guarded probe so the
    # underlying (possibly shelling-out) command runs at most once.
    gpu_raw: dict = {}
    if time.monotonic() < deadline:
      gpu_raw = self._safe_gpu()
    gpu_model = self._safe(lambda: gpu_raw.get("model"))
    total_vram_bytes = self._normalize_vram(gpu_raw.get("vram_bytes"))
    accelerator = self._safe(lambda: gpu_raw.get("accelerator"))

    static = StaticMetrics(
      cpu_model=cpu_model,
      gpu_model=gpu_model,
      accelerator=accelerator,
      logical_cores=logical_cores,
      physical_cores=physical_cores,
      total_ram_bytes=total_ram_bytes,
      cpu_architecture=cpu_architecture,
      total_vram_bytes=total_vram_bytes,
    )

    os_metrics = OSMetrics(
      os_name=os_name,
      os_version=os_version,
      python_version=python_version,
      machine_architecture=machine_architecture,
    )

    return SystemProfile(static=static, os=os_metrics)

  # ---------------------------------------------------------------------
  # Persistence (JSON load / write)
  # ---------------------------------------------------------------------

  # Serialization schema version for the System_Info_File (Requirement 3.2).
  _SCHEMA_VERSION = 1

  def _read_file(self) -> Optional[SystemProfile]:
    """Read and JSON-parse the ``System_Info_File`` into a ``SystemProfile``.

    Returns ``None`` when the file is missing, unreadable, or unparseable
    (invalid JSON or not a JSON object), which drives the collect-and-write
    branch (Requirements 4.1, 4.3). Unknown keys are ignored and any absent
    field defaults to ``UNKNOWN``, so a successful JSON decode always yields a
    total profile with an entry for every defined field.
    """
    try:
      with open(self._system_info_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    except (OSError, ValueError):
      # Missing, unreadable, or invalid JSON -> treat as unparseable.
      return None

    if not isinstance(data, dict):
      return None

    static_raw = data.get("static")
    os_raw = data.get("os")
    if not isinstance(static_raw, dict):
      static_raw = {}
    if not isinstance(os_raw, dict):
      os_raw = {}

    def _string_field(source: dict, name: str) -> str:
      # Absent (or non-string) fields default to UNKNOWN so deserialization
      # always yields a total profile.
      value = source.get(name, UNKNOWN)
      if not isinstance(value, str):
        return UNKNOWN
      return value

    static = StaticMetrics(
      **{f.name: _string_field(static_raw, f.name) for f in fields(StaticMetrics)}
    )
    os_metrics = OSMetrics(
      **{f.name: _string_field(os_raw, f.name) for f in fields(OSMetrics)}
    )
    return SystemProfile(static=static, os=os_metrics)

  def _write_file(self, profile: SystemProfile) -> None:
    """Serialize ``profile`` to JSON and write it to the ``System_Info_File``.

    The full profile is serialized — including any ``UNKNOWN`` fields — with a
    version key and one entry per metric group/field, so the file always
    contains every defined Static and OS metric (Requirement 3.2). The
    ``data_dir`` is created if it does not yet exist. On any I/O failure the
    error is logged and the method returns without raising, leaving the
    in-memory profile intact (Requirement 3.3).
    """
    document = {
      "version": self._SCHEMA_VERSION,
      "static": {f.name: getattr(profile.static, f.name) for f in fields(StaticMetrics)},
      "os": {f.name: getattr(profile.os, f.name) for f in fields(OSMetrics)},
    }
    try:
      if self._data_dir:
        os.makedirs(self._data_dir, exist_ok=True)
      with open(self._system_info_file, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    except OSError:
      logger.warning(
        "Failed to write System_Info_File at %s; keeping profile in memory",
        self._system_info_file,
        exc_info=True,
      )

  # ---------------------------------------------------------------------
  # Profile resolution API
  # ---------------------------------------------------------------------

  def load_cached_profile(self) -> bool:
    """ Load the persisted profile into memory without collecting (fast) """
    try:
      loaded = self._read_file()
    except Exception:
      logger.warning("Failed to read cached system profile", exc_info=True)
      loaded = None
    if loaded is not None:
      self._profile = loaded
      return True
    if self._profile is None:
      # No usable cache: seed a placeholder so nothing blocks on a synchronous
      # collection; refresh() will overwrite it with real data shortly.
      self._profile = SystemProfile(static=StaticMetrics(), os=OSMetrics())
    return False

  def refresh(self) -> SystemProfile:
    """ Collect a fresh profile, cache it in memory, and persist it to disk """
    try:
      profile = self._collect()
      self._profile = profile
      self._write_file(profile)
      return profile
    except Exception:
      logger.error(
        "Unexpected error while refreshing system profile; "
        "keeping any previously cached profile",
        exc_info=True,
      )
      if self._profile is None:
        self._profile = SystemProfile(static=StaticMetrics(), os=OSMetrics())
      return self._profile

  def ensure_profile(self) -> None:
    """Resolve the in-memory ``SystemProfile`` exactly once.

    If a profile is already cached in memory, returns immediately without any
    file access (Requirement 4.4). Otherwise it consults the
    ``System_Info_File`` via ``_read_file()``:

      * when the file exists and parses, the loaded profile is cached and no
        collection is performed (load path, Requirement 4.1);
      * when the file is missing *or* present but unreadable/unparseable,
        ``_read_file()`` returns ``None``; the service then ``_collect()``s a
        fresh profile, caches it, and ``_write_file()``s it — writing a fresh
        file when absent (Requirement 4.2) and overwriting the bad file when
        present (Requirement 4.3).

    The whole body is guarded so it never raises: an unexpected internal error
    caches a profile of all-``UNKNOWN`` values and logs (Error Handling §5).
    """
    if self._profile is not None:
      return
    try:
      loaded = self._read_file()
      if loaded is not None:
        # Load path: reuse the persisted profile without collecting.
        self._profile = loaded
        return
      # Missing or unreadable/unparseable: collect, cache, and persist. A
      # single collect+write branch covers both cases because _read_file()
      # returns None for each.
      profile = self._collect()
      self._profile = profile
      self._write_file(profile)
    except Exception:
      logger.error(
        "Unexpected error while resolving system profile; "
        "caching a profile of unknown values",
        exc_info=True,
      )
      self._profile = SystemProfile(static=StaticMetrics(), os=OSMetrics())

  def get_profile(self) -> SystemProfile:
    """Return the full ``SystemProfile``, resolving it once if needed.

    Calls ``ensure_profile()`` when no profile is cached, then returns the
    cached profile. Once a profile is in memory it is reused on every
    subsequent call without re-reading the ``System_Info_File`` (Requirement
    4.4). Never raises: if resolution somehow leaves no cached profile, an
    all-``UNKNOWN`` profile is returned.
    """
    if self._profile is None:
      self.ensure_profile()
    if self._profile is None:
      # Defensive: ensure_profile always caches a profile, but never leak a
      # None out of the public API.
      return SystemProfile(static=StaticMetrics(), os=OSMetrics())
    return self._profile

  # ---------------------------------------------------------------------
  # Ambient context formatting
  # ---------------------------------------------------------------------

  @staticmethod
  def _format_bytes_as_gb(raw: str) -> str:
    """Render a byte-valued metric as a gigabyte figure.

    Converts using decimal GB (``bytes / 1_000_000_000``, matching how RAM and
    VRAM are marketed and how users reason about model sizes), rounded to one
    decimal place with a ``GB`` unit, e.g. ``"34.4 GB"`` (Requirement 6.3). A
    byte field whose value is ``UNKNOWN`` — or is otherwise missing or
    non-numeric — is rendered literally as ``unknown`` with no unit appended.
    """
    if raw == UNKNOWN:
      return UNKNOWN
    try:
      value = int(str(raw).strip())
    except (TypeError, ValueError):
      return UNKNOWN
    gb = value / 1_000_000_000
    return f"{gb:.1f} GB"

  def format_ambient_context(self) -> str:
    """Render the ``SystemProfile`` into a human-readable context block.

    Produces a ``<system_information>`` block that presents every Static and OS
    metric field in human-readable text (Requirement 6.2). Byte-valued fields
    (total RAM, total VRAM) are expressed in GB via ``_format_bytes_as_gb``
    (Requirement 6.3); every other field is rendered verbatim, and any
    ``UNKNOWN`` value is shown literally so the agent knows the metric was
    attempted rather than omitted. The whole body is guarded so it never raises
    — any unexpected error logs and returns an empty string (Error Handling §5).
    """
    try:
      profile = self.get_profile()
      static = profile.static
      os_metrics = profile.os
      lines = [
        "<system_information>",
        "The following describes the machine this assistant is running on. Use it to reason about",
        "which local models can run and to give system-aware answers.",
        "",
        f"Operating System: {os_metrics.os_name} (version {os_metrics.os_version})",
        f"Architecture: {os_metrics.machine_architecture} / {static.cpu_architecture}",
        f"Python Runtime: {os_metrics.python_version}",
        "",
        (
          f"CPU: {static.cpu_model} "
          f"({static.physical_cores} physical cores, {static.logical_cores} logical cores)"
        ),
        f"Total RAM: {self._format_bytes_as_gb(static.total_ram_bytes)}",
        f"GPU: {static.gpu_model}",
        f"Total VRAM: {self._format_bytes_as_gb(static.total_vram_bytes)}",
        f"Accelerator: {static.accelerator}",
        "</system_information>",
      ]
      return "\n".join(lines)
    except Exception:
      logger.warning(
        "Failed to format ambient context; returning empty string",
        exc_info=True,
      )
      return ""

  # ---------------------------------------------------------------------
  # Optional-library and shell helpers
  # ---------------------------------------------------------------------

  def _get_psutil(self) -> Any:
    """Lazily import the optional ``psutil`` inspection library.

    Returns the module when importable, or ``None`` when it is unavailable so
    callers fall back to the standard library (Requirement 5.3). Isolated in
    its own method to make the "library absent" path easy to exercise in tests.
    """
    try:
      import psutil  # type: ignore
      return psutil
    except Exception:
      return None

  def _run_command(self, args: list, timeout: Optional[float] = PROBE_TIMEOUT_SECONDS) -> str:
    """Run an external command and return its trimmed stdout.

    Returns an empty string when the executable is not on PATH or the command
    exits non-zero. Any timeout or OS error propagates to the enclosing guard
    (``_safe`` / ``_safe_gpu``), which records ``UNKNOWN``.
    """
    if not args or shutil.which(args[0]) is None:
      return ""
    result = subprocess.run(
      args,
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
    )
    if result.returncode != 0:
      return ""
    return (result.stdout or "").strip()

  # ---------------------------------------------------------------------
  # Static metric probes
  # ---------------------------------------------------------------------

  def _probe_cpu_model(self) -> Any:
    """Best-effort human-readable CPU model name (no PII)."""
    system = platform.system()
    if system == "Linux":
      try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
          for line in handle:
            if line.lower().startswith("model name"):
              return line.split(":", 1)[1].strip()
      except OSError:
        pass
      return platform.processor() or platform.machine()
    if system == "Darwin":
      brand = self._run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
      if brand:
        return brand
      return platform.processor() or platform.machine()
    if system == "Windows":
      # PROCESSOR_IDENTIFIER describes the CPU family, not the user/host.
      return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    return platform.processor() or platform.machine()

  def _probe_physical_cores(self) -> Any:
    """Physical core count. Requires ``psutil``; no portable stdlib source."""
    psutil = self._get_psutil()
    if psutil is not None:
      return psutil.cpu_count(logical=False)
    return None

  def _probe_logical_cores(self) -> Any:
    """Logical core count via ``psutil`` with stdlib fallbacks."""
    psutil = self._get_psutil()
    if psutil is not None:
      count = psutil.cpu_count(logical=True)
      if count:
        return count
    count = os.cpu_count()
    if count:
      return count
    try:
      return multiprocessing.cpu_count()
    except (NotImplementedError, ValueError):
      return None

  def _probe_total_ram_bytes(self) -> Any:
    """Total installed RAM in bytes via ``psutil`` or platform fallbacks."""
    psutil = self._get_psutil()
    if psutil is not None:
      return psutil.virtual_memory().total
    return self._probe_total_ram_bytes_stdlib()

  def _probe_total_ram_bytes_stdlib(self) -> Any:
    """Standard-library / platform fallback for total RAM in bytes."""
    system = platform.system()
    if system == "Linux":
      try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as handle:
          for line in handle:
            if line.startswith("MemTotal:"):
              kib = int(line.split()[1])
              return kib * 1024
      except (OSError, ValueError, IndexError):
        return None
      return None
    if system == "Darwin":
      out = self._run_command(["sysctl", "-n", "hw.memsize"])
      return int(out) if out.isdigit() else None
    if system == "Windows":
      return self._windows_total_ram_bytes()
    return None

  def _windows_total_ram_bytes(self) -> Any:
    """Total physical memory on Windows via ``GlobalMemoryStatusEx``."""
    import ctypes

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
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
      return None
    return stat.ullTotalPhys

  def _probe_cpu_architecture(self) -> Any:
    """CPU / instruction-set architecture (e.g. ``x86_64``, ``arm64``)."""
    return platform.machine()

  # ---------------------------------------------------------------------
  # GPU / VRAM / accelerator probes (best-effort, platform-dispatched)
  # ---------------------------------------------------------------------

  def _safe_gpu(self) -> dict:
    """Run the platform GPU probe, swallowing any failure.

    Returns a dict that may contain ``model``, ``vram_bytes`` and
    ``accelerator`` keys, or an empty dict when nothing could be determined.
    Never raises (Requirement 5.2).
    """
    try:
      result = self._probe_gpu()
      return result or {}
    except Exception:
      logger.warning("GPU probe failed; recording %r", UNKNOWN, exc_info=True)
      return {}

  def _probe_gpu(self) -> dict:
    """Dispatch GPU detection to the appropriate platform probe."""
    system = platform.system()
    if system == "Windows":
      return self._probe_gpu_windows()
    if system == "Darwin":
      return self._probe_gpu_macos()
    if system == "Linux":
      return self._probe_gpu_linux()
    return {}

  def _probe_gpu_nvidia(self) -> dict:
    """Query an NVIDIA GPU via ``nvidia-smi`` (Windows and Linux)."""
    out = self._run_command(
      ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
      return {}
    first = out.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    result: dict = {"accelerator": "CUDA"}
    if parts and parts[0]:
      result["model"] = parts[0]
    if len(parts) > 1 and parts[1]:
      try:
        mib = float(parts[1])
        result["vram_bytes"] = int(mib * 1024 * 1024)
      except ValueError:
        pass
    return result

  def _probe_gpu_windows(self) -> dict:
    """GPU detection on Windows: NVIDIA first, then WMI ``win32_VideoController``."""
    info = self._probe_gpu_nvidia()
    if info.get("model"):
      return info
    name = self._run_command(
      ["wmic", "path", "win32_VideoController", "get", "name"]
    )
    model = self._first_value_line(name, header="name")
    ram = self._run_command(
      ["wmic", "path", "win32_VideoController", "get", "AdapterRAM"]
    )
    vram = self._first_value_line(ram, header="adapterram")
    result: dict = {}
    if model:
      result["model"] = model
    if vram and vram.isdigit():
      result["vram_bytes"] = int(vram)
    return result or info

  def _probe_gpu_macos(self) -> dict:
    """GPU detection on macOS via ``system_profiler SPDisplaysDataType``."""
    out = self._run_command(["system_profiler", "SPDisplaysDataType"])
    result: dict = {}
    for line in out.splitlines():
      stripped = line.strip()
      if stripped.startswith("Chipset Model:") and "model" not in result:
        result["model"] = stripped.split(":", 1)[1].strip()
      elif stripped.startswith("VRAM") and "vram_bytes" not in result:
        size = self._parse_size_to_bytes(stripped.split(":", 1)[1].strip())
        if size:
          result["vram_bytes"] = size
    # Apple Silicon exposes the Apple Neural Engine accelerator.
    if platform.machine() == "arm64":
      result.setdefault("accelerator", "Apple Neural Engine")
    return result

  def _probe_gpu_linux(self) -> dict:
    """GPU detection on Linux: NVIDIA first, then ``lspci`` for a model name."""
    info = self._probe_gpu_nvidia()
    if info.get("model"):
      return info
    out = self._run_command(["lspci"])
    for line in out.splitlines():
      lowered = line.lower()
      if "vga compatible controller" in lowered or "3d controller" in lowered:
        return {"model": line.split(":", 2)[-1].strip()}
    return info

  @staticmethod
  def _first_value_line(text: str, header: str) -> str:
    """Return the first non-empty, non-header line of a ``wmic`` table."""
    for line in (text or "").splitlines():
      stripped = line.strip()
      if not stripped or stripped.lower() == header.lower():
        continue
      return stripped
    return ""

  @staticmethod
  def _parse_size_to_bytes(text: str) -> Optional[int]:
    """Parse a size like ``"4 GB"`` or ``"1536 MB"`` into bytes."""
    match = re.match(r"([\d.]+)\s*(GB|MB|KB|B)?", text or "", re.IGNORECASE)
    if not match:
      return None
    try:
      number = float(match.group(1))
    except ValueError:
      return None
    unit = (match.group(2) or "B").upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    return int(number * multipliers[unit])

  # ---------------------------------------------------------------------
  # OS / runtime metric probes
  # ---------------------------------------------------------------------

  def _probe_os_name(self) -> Any:
    """Human-readable operating-system name including edition where available.
    """
    system = platform.system()
    if system == "Windows":
      # Prefer the in-process edition lookup (no subprocess): platform exposes
      # the release ("11") and edition ("Professional" -> "Pro"), which is both
      # fast and reliable. Only shell out for the WMI caption as a fallback.
      release = platform.release()
      edition = self._windows_edition()
      if release and edition:
        return f"{system} {release} {edition}".strip()
      caption = self._windows_os_caption()
      if caption:
        return caption
      base = f"{system} {release}".strip()
      return f"{base} {edition}".strip() if edition else base
    release = platform.release()
    if system and release:
      return f"{system} {release}"
    return system or release

  @staticmethod
  def _windows_edition() -> str:
    """Return the Windows edition (e.g. ``Pro``) via ``platform.win32_edition``."""
    try:
      edition = platform.win32_edition()  # type: ignore[attr-defined]
    except Exception:
      return ""
    if not edition:
      return ""
    # win32_edition returns registry-style names like "Professional" or "Core".
    mapping = {
      "Professional": "Pro",
      "Core": "Home",
      "CoreSingleLanguage": "Home Single Language",
      "Enterprise": "Enterprise",
      "Education": "Education",
    }
    return mapping.get(edition, edition)

  def _windows_os_caption(self) -> str:
    """Return the full Windows product caption (e.g. ``Windows 11 Pro``).

    Tries PowerShell/CIM first and falls back to ``wmic``; the leading
    ``Microsoft`` vendor prefix is stripped so the name reads naturally.
    """
    caption = self._run_command([
      "powershell", "-NoProfile", "-Command",
      "(Get-CimInstance Win32_OperatingSystem).Caption",
    ])
    if not caption:
      raw = self._run_command(["wmic", "os", "get", "Caption"])
      caption = self._first_value_line(raw, header="caption")
    caption = caption.strip()
    if caption.lower().startswith("microsoft "):
      caption = caption[len("microsoft "):].strip()
    return caption

  def _probe_os_version(self) -> Any:
    """Operating-system version string."""
    return platform.version()

  def _probe_machine_architecture(self) -> Any:
    """Machine architecture as reported by the OS (e.g. ``AMD64``)."""
    return platform.machine()

  def _probe_python_version(self) -> Any:
    """Python runtime version."""
    version = platform.python_version()
    if version:
      return version
    return sys.version.split()[0] if sys.version else None
