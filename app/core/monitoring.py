"""Service-health probing and Docker stats collection for the monitoring page.

This module talks to two surfaces:

* the per-service HTTP ``/health`` endpoints (FastAPI services + Streamlit
  ``/_stcore/health``) plus a TCP/SQL probe for the bare Postgres container, to
  classify each service as ``up``/``down`` with a latency reading;
* the Docker Engine API over the bind-mounted UNIX socket
  (``/var/run/docker.sock``), to read per-container CPU%, memory usage, and
  network throughput.

Both are designed to be cheap (small timeouts, single-shot per refresh) and
fail-soft — a missing socket or a downed container yields a row with the
problem surfaced, never a page-level exception.
"""

from __future__ import annotations

import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")
DOCKER_SOCKET = "/var/run/docker.sock"

HEALTH_PROBE_TIMEOUT = 2.0
DOCKER_STATS_TIMEOUT = 4.0


@dataclass
class HealthResult:
    """Result of one service health probe.

    Args:
        service: Logical service name (``agent``, ``db``, ...).
        url: Endpoint actually probed, for display.
        status: One of ``up``, ``down``, ``skipped``.
        latency_ms: Round-trip latency in milliseconds, or ``None`` on failure.
        detail: Optional explanatory string (e.g. error text).
    """

    service: str
    url: str
    status: str
    latency_ms: float | None
    detail: str


@dataclass
class ContainerStats:
    """Snapshot of one container's resource use from ``/containers/.../stats``.

    Args:
        name: Container name as reported by Docker.
        cpu_percent: Total CPU usage as a percentage of all cores.
        memory_used_mb: Resident memory in MB.
        memory_limit_mb: Per-container memory limit in MB.
        memory_percent: ``memory_used / memory_limit * 100``.
        rx_mb: Cumulative network bytes received, in MB.
        tx_mb: Cumulative network bytes transmitted, in MB.
        status: Docker container status string (``running``, ``exited``...).
    """

    name: str
    cpu_percent: float
    memory_used_mb: float
    memory_limit_mb: float
    memory_percent: float
    rx_mb: float
    tx_mb: float
    status: str


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _http_targets() -> list[tuple[str, str]]:
    """Return ``[(service_name, health_url), ...]`` for every HTTP-probed service."""
    cfg = _load_config()

    def port(section: str, default: int) -> int:
        return int((cfg.get(section) or {}).get("port", default))

    qdrant_host = (cfg.get("qdrant") or {}).get("host", "vector_db")
    qdrant_port = (cfg.get("qdrant") or {}).get("port", 6333)

    return [
        ("agent", f"http://agent:{port('agent', 8000)}/health"),
        ("forecaster", f"http://forecaster:{port('forecaster', 8001)}/health"),
        ("clustering", f"http://clustering:{port('clustering', 8002)}/health"),
        ("downloader_extra", f"http://downloader_extra:{port('downloader_extra', 8003)}/health"),
        ("python_sandbox", f"http://python_sandbox:{port('python_sandbox', 8004)}/health"),
        ("app", f"http://app:{port('app', 8501)}/_stcore/health"),
        ("vector_db", f"http://{qdrant_host}:{qdrant_port}/readyz"),
    ]


def _probe_http(service: str, url: str) -> HealthResult:
    start = _monotonic_ms()
    try:
        with httpx.Client(timeout=HEALTH_PROBE_TIMEOUT) as client:
            response = client.get(url)
        latency = _monotonic_ms() - start
        status = "up" if response.status_code == 200 else "down"
        return HealthResult(
            service=service,
            url=url,
            status=status,
            latency_ms=latency,
            detail=f"HTTP {response.status_code}",
        )
    except httpx.HTTPError as exc:
        return HealthResult(
            service=service,
            url=url,
            status="down",
            latency_ms=None,
            detail=str(exc).strip() or exc.__class__.__name__,
        )


def _probe_postgres() -> HealthResult:
    cfg = _load_config().get("postgres") or {}
    host = str(cfg.get("host", "db"))
    port = int(cfg.get("port", 5432))
    start = _monotonic_ms()
    try:
        with socket.create_connection((host, port), timeout=HEALTH_PROBE_TIMEOUT):
            pass
        return HealthResult(
            service="db",
            url=f"tcp://{host}:{port}",
            status="up",
            latency_ms=_monotonic_ms() - start,
            detail="TCP accept",
        )
    except OSError as exc:
        return HealthResult(
            service="db",
            url=f"tcp://{host}:{port}",
            status="down",
            latency_ms=None,
            detail=str(exc).strip() or exc.__class__.__name__,
        )


def probe_all_services() -> list[HealthResult]:
    """Probe every known service in parallel and return per-service results."""
    targets = _http_targets()
    results: list[HealthResult] = []
    with ThreadPoolExecutor(max_workers=min(8, len(targets) + 1)) as executor:
        futures = [executor.submit(_probe_http, name, url) for name, url in targets]
        futures.append(executor.submit(_probe_postgres))
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                logger.warning("Health probe future raised: %s", exc)
    results.sort(key=lambda r: r.service)
    return results


def _docker_client() -> httpx.Client | None:
    """Build an httpx client speaking to the Docker socket; ``None`` if unavailable."""
    if not os.path.exists(DOCKER_SOCKET):
        return None
    transport = httpx.HTTPTransport(uds=DOCKER_SOCKET)
    return httpx.Client(
        transport=transport, base_url="http://localhost", timeout=DOCKER_STATS_TIMEOUT
    )


def _cpu_percent(stats: dict[str, Any]) -> float:
    """Compute total CPU% across all cores from a single ``/stats?stream=false`` payload."""
    cpu = stats.get("cpu_stats") or {}
    pre = stats.get("precpu_stats") or {}
    try:
        cpu_total = cpu["cpu_usage"]["total_usage"]
        pre_total = pre["cpu_usage"]["total_usage"]
        sys_total = cpu.get("system_cpu_usage", 0) or 0
        pre_sys = pre.get("system_cpu_usage", 0) or 0
        cpu_delta = cpu_total - pre_total
        sys_delta = sys_total - pre_sys
        online_cpus = cpu.get("online_cpus") or len(
            (cpu.get("cpu_usage") or {}).get("percpu_usage") or [1]
        )
    except (KeyError, TypeError):
        return 0.0
    if sys_delta <= 0 or cpu_delta <= 0:
        return 0.0
    return (cpu_delta / sys_delta) * online_cpus * 100.0


def _net_bytes(stats: dict[str, Any]) -> tuple[float, float]:
    networks = stats.get("networks") or {}
    rx = sum(int(v.get("rx_bytes", 0) or 0) for v in networks.values())
    tx = sum(int(v.get("tx_bytes", 0) or 0) for v in networks.values())
    return rx / 1024 / 1024, tx / 1024 / 1024


def _fetch_one_container_stats(container: dict[str, Any]) -> ContainerStats | None:
    """Fetch a single container's stats snapshot using a fresh socket client.

    A dedicated client per worker is required because ``httpx.Client`` is not
    thread-safe for concurrent requests and the Docker stats endpoint blocks
    for ~1s waiting for the second CPU sample. Returning ``None`` lets the
    caller drop unreadable containers without aborting the whole refresh.
    """
    container_id = container.get("Id")
    name = (container.get("Names") or [""])[0].lstrip("/")
    container_status = container.get("State", "unknown")
    if not container_id:
        return None

    client = _docker_client()
    if client is None:
        return None
    try:
        stats = client.get(f"/containers/{container_id}/stats", params={"stream": "false"}).json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Stats fetch failed for %s: %s", name, exc)
        return None
    finally:
        client.close()

    memory = stats.get("memory_stats") or {}
    mem_used = int(memory.get("usage", 0) or 0) / 1024 / 1024
    mem_limit = int(memory.get("limit", 0) or 0) / 1024 / 1024
    mem_pct = (mem_used / mem_limit * 100.0) if mem_limit > 0 else 0.0
    rx_mb, tx_mb = _net_bytes(stats)
    return ContainerStats(
        name=name,
        cpu_percent=_cpu_percent(stats),
        memory_used_mb=mem_used,
        memory_limit_mb=mem_limit,
        memory_percent=mem_pct,
        rx_mb=rx_mb,
        tx_mb=tx_mb,
        status=container_status,
    )


def get_container_stats() -> list[ContainerStats]:
    """Return one :class:`ContainerStats` per Compose container, or an empty list.

    The Docker stats endpoint blocks ~1s per container to compute the second
    CPU sample. Per-container fetches run in parallel so the total wait stays
    roughly constant in the number of containers.
    """
    client = _docker_client()
    if client is None:
        return []

    try:
        containers = client.get(
            "/containers/json",
            params={"all": "true", "filters": '{"label":["com.docker.compose.project"]}'},
        ).json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Failed to list containers from docker socket: %s", exc)
        return []
    finally:
        client.close()

    if not containers:
        return []

    stats_rows: list[ContainerStats] = []
    with ThreadPoolExecutor(max_workers=min(16, len(containers))) as executor:
        futures = [executor.submit(_fetch_one_container_stats, c) for c in containers]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("Container stats future raised: %s", exc)
                continue
            if result is not None:
                stats_rows.append(result)

    stats_rows.sort(key=lambda c: c.name)
    return stats_rows


def docker_socket_available() -> bool:
    """Return ``True`` iff the Docker socket is mounted at the expected path."""
    return os.path.exists(DOCKER_SOCKET)


def _monotonic_ms() -> float:
    import time

    return time.monotonic() * 1000.0
