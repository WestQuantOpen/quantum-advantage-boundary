"""Clean adapter around the official BlueQubit Python SDK.

The BlueQubit SDK (v0.18.12b1) exposes:
- bluequbit.init(api_token=None, execution_mode=None) -> BQClient
- BQClient.run(circuits, device='cpu', asynchronous=False, job_name=None, shots=None, pauli_sum=None, options=None, tags=None, timeout=None, force_run=False) -> JobResult
- BQClient.estimate(circuits, device='cpu') -> EstimateResult
- BQClient.wait(job_ids, timeout=None) -> JobResult
- BQClient.cancel(job_ids) -> JobResult
- BQClient.search(run_status=None, created_later_than=None, batch_id=None) -> list[JobResult]
- BQClient.get(job_ids) -> JobResult
- JobResult.get_counts(), JobResult.get_statevector(), JobResult.ok
- JobResult has attributes: job_id, run_time_seconds, cost, shots, device, run_status

There is NO public BlueQubit AI-agent API. We use this SDK strictly as the compute layer.
"""

from __future__ import annotations

import csv
import datetime
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import bluequbit
from bluequbit.exceptions import (
    BQAPIError,
    BQError,
    BQJobInvalidDeviceTypeError,
    BQJobsMalformedShotsError,
    BQSDKUsageError,
    BQUnauthorizedAccessError,
)

logger = structlog.get_logger(__name__)

# Errors that are worth retrying (transient / network / server-side).
# Non-transient errors (auth, usage, validation) are excluded so we fail fast.
_TRANSIENT_ERRORS = (BQAPIError, ConnectionError, TimeoutError, OSError)

# CSV ledger header.  ``timestamp`` is appended for auditability.
_LEDGER_HEADER: list[str] = [
    "experiment_id",
    "backend",
    "qubits",
    "shots",
    "estimated_cost",
    "actual_cost",
    "estimated_runtime",
    "actual_runtime",
    "status",
    "agent",
    "reason",
    "timestamp",
]


def _before_sleep_log(retry_state: Any) -> None:
    """Log a retry attempt using structlog (never logs credentials)."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "bluequbit.retry",
        attempt=retry_state.attempt_number,
        wait=getattr(retry_state, "next_action", None),
        error=repr(exc) if exc else None,
    )


def _retry_policy() -> Any:
    """Build a fresh tenacity retry decorator instance.

    Returns a decorator configured for at most 3 attempts with exponential
    backoff (1s -> 2s -> 4s, capped at 10s) on transient errors only.
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        before_sleep=_before_sleep_log,
        reraise=True,
    )


def _mask_token(token: str | None) -> str:
    """Return a masked representation of the API token for safe logging."""
    if not token:
        return "<unset>"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


class BlueQubitAdapter:
    """Thin, safe wrapper around the BlueQubit compute SDK.

    Responsibilities:
        * Run quantum circuits (single + batch) on BlueQubit backends.
        * Estimate cost / runtime before running.
        * Select an appropriate device based on circuit width.
        * Record every job in a durable CSV ledger for budget enforcement.
        * Enforce a hard cost ceiling (``max_cost_usd``).

    The adapter NEVER logs the raw API token.  It only ever records a masked
    prefix/suffix for diagnostic purposes.
    """

    def __init__(self, api_token: str | None = None, max_cost_usd: float = 10.0) -> None:
        self._lock = threading.Lock()
        self.max_cost_usd = float(max_cost_usd)

        # Resolve the API token: explicit arg takes priority, then env var.
        token = api_token or os.environ.get("BLUEQUBIT_API_TOKEN")
        self._api_token = token

        # Ledger path is anchored to the project root (parent of ``src/``).
        project_root = Path(__file__).resolve().parent.parent
        self.ledger_path = project_root / "state" / "compute_ledger.csv"

        # Initialise the underlying BlueQubit client.  ``init`` will itself fall
        # back to the ``BLUEQUBIT_API_TOKEN`` env var when ``api_token`` is None.
        # The SDK authenticates eagerly, so a missing/invalid token raises here.
        # We degrade gracefully: ledger / budget / device-selection still work
        # without a live connection; compute methods raise a clear error.
        self._client: Any = None
        try:
            self._client = bluequbit.init(api_token=token)
            logger.info(
                "bluequbit.adapter.initialised",
                token=_mask_token(token),
                max_cost_usd=self.max_cost_usd,
                ledger=str(self.ledger_path),
            )
        except Exception as exc:
            logger.warning(
                "bluequbit.init_failed",
                token=_mask_token(token),
                error=repr(exc),
            )
            logger.warning(
                "bluequbit.adapter.degraded",
                reason="compute methods unavailable; ledger/budget still work",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Any,
        device: str = "cpu",
        shots: int = 1000,
        job_name: str | None = None,
        tags: dict[str, Any] | None = None,
        asynchronous: bool = False,
    ) -> dict[str, Any]:
        """Run a single circuit and return a result dictionary.

        Returns a dict with keys: ``counts``, ``job_id``, ``cost``,
        ``runtime``, ``device``.
        """
        if self.budget_exceeded:
            logger.warning("bluequbit.budget_exceeded", total_cost=self.total_cost)
            raise RuntimeError(
                f"Compute budget exceeded: ${self.total_cost:.4f} >= "
                f"${self.max_cost_usd:.4f}"
            )

        logger.info(
            "bluequbit.run",
            device=device,
            shots=shots,
            job_name=job_name,
            asynchronous=asynchronous,
        )

        result = self._run_with_retry(
            circuit,
            device=device,
            shots=shots,
            job_name=job_name,
            tags=tags,
            asynchronous=asynchronous,
        )

        if asynchronous:
            # Only job_id / status available until completion.
            return self._job_result_to_dict(result, include_counts=False)

        return self._job_result_to_dict(result, include_counts=True)

    def run_batch(
        self,
        circuits: Sequence[Any],
        device: str = "cpu",
        shots: int = 1000,
        job_names: Sequence[str] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run multiple circuits asynchronously, wait for all, return results.

        All circuits are submitted in a single batch with ``asynchronous=True``
        and then awaited via ``BQClient.wait``.  Returns a list of result dicts
        in the same order as ``circuits``.
        """
        if self.budget_exceeded:
            raise RuntimeError(
                f"Compute budget exceeded: ${self.total_cost:.4f} >= "
                f"${self.max_cost_usd:.4f}"
            )

        circuits = list(circuits)
        if not circuits:
            return []

        logger.info("bluequbit.run_batch", n=len(circuits), device=device, shots=shots)

        # Submit asynchronously.
        job_results = self._run_with_retry(
            circuits,
            device=device,
            shots=shots,
            job_names=job_names,
            tags=tags,
            asynchronous=True,
        )

        # Normalise to a list.
        if not isinstance(job_results, list):
            job_results = [job_results]

        # Wait for every job to finish.
        completed = self._wait_with_retry(job_results)
        if not isinstance(completed, list):
            completed = [completed]

        return [self._job_result_to_dict(r, include_counts=True) for r in completed]

    def estimate(
        self, circuit: Any, device: str = "cpu"
    ) -> dict[str, Any]:
        """Return an estimated cost / runtime for a circuit.

        Returns a dict with keys: ``estimated_cost``, ``estimated_runtime``
        (seconds), ``device``, ``num_qubits``.
        """
        logger.info("bluequbit.estimate", device=device)
        est = self._estimate_with_retry(circuit, device=device)
        if isinstance(est, list):
            est = est[0]

        runtime_ms = getattr(est, "estimated_runtime", None)
        return {
            "estimated_cost": getattr(est, "estimated_cost", None),
            "estimated_runtime": (runtime_ms / 1000.0) if runtime_ms is not None else None,
            "device": getattr(est, "device", device),
            "num_qubits": getattr(est, "num_qubits", None),
            "warning_message": getattr(est, "warning_message", None),
            "error_message": getattr(est, "error_message", None),
        }

    def select_device(self, n_qubits: int, prefer_gpu: bool = False) -> str:
        """Select a backend based on circuit width.

        Heuristic:
            * n <= 20  -> ``cpu``  (statevector fits comfortably)
            * n <= 26  -> ``gpu``  (needs GPU memory)
            * n  > 26  -> ``mps.cpu`` (tensor-network / MPS required)

        When ``prefer_gpu`` is True, GPU is preferred for any circuit that
        fits (``n <= 26``).
        """
        if prefer_gpu and n_qubits <= 26:
            device = "gpu"
        elif n_qubits <= 20:
            device = "cpu"
        elif n_qubits <= 26:
            device = "gpu"
        else:
            device = "mps.cpu"

        logger.info("bluequbit.select_device", n_qubits=n_qubits, prefer_gpu=prefer_gpu, device=device)
        return device

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------
    def record_to_ledger(
        self,
        experiment_id: str,
        backend: str,
        qubits: int,
        shots: int,
        estimated_cost: float | None,
        actual_cost: float | None,
        estimated_runtime: float | None,
        actual_runtime: float | None,
        status: str,
        agent: str,
        reason: str,
    ) -> None:
        """Append a single row to the compute ledger CSV.

        The ledger is created with a header row if it does not yet exist.
        Thread-safe via an internal lock.
        """
        row = [
            experiment_id,
            backend,
            qubits,
            shots,
            estimated_cost,
            actual_cost,
            estimated_runtime,
            actual_runtime,
            status,
            agent,
            reason,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ]

        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not self.ledger_path.exists()
            with open(self.ledger_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                if needs_header:
                    writer.writerow(_LEDGER_HEADER)
                writer.writerow(row)

        logger.info(
            "bluequbit.ledger.recorded",
            experiment_id=experiment_id,
            backend=backend,
            actual_cost=actual_cost,
            status=status,
        )

    @property
    def total_cost(self) -> float:
        """Sum of all *actual* costs recorded in the ledger."""
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        with self._lock:
            with open(self.ledger_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    val = row.get("actual_cost")
                    if val is None or val == "" or val == "None":
                        continue
                    try:
                        total += float(val)
                    except (TypeError, ValueError):
                        continue
        return total

    @property
    def budget_exceeded(self) -> bool:
        """True when cumulative actual cost meets or exceeds the budget."""
        return self.total_cost >= self.max_cost_usd

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_client(self) -> Any:
        """Return the live BlueQubit client or raise a clear error."""
        if self._client is None:
            raise RuntimeError(
                "BlueQubit client is not initialised (no valid API token). "
                "Set BLUEQUBIT_API_TOKEN or pass a valid api_token."
            )
        return self._client

    def _run_with_retry(
        self,
        circuit: Any,
        device: str = "cpu",
        shots: int = 1000,
        job_name: str | None = None,
        job_names: Sequence[str] | None = None,
        tags: dict[str, Any] | None = None,
        asynchronous: bool = False,
    ) -> Any:
        """Submit a job to BlueQubit with tenacity-managed retries."""
        client = self._require_client()

        @_retry_policy()
        def _submit() -> Any:
            # For batch submissions, pass the list directly.
            if isinstance(circuit, list):
                return client.run(
                    circuit,
                    device=device,
                    asynchronous=asynchronous,
                    shots=shots,
                    tags=tags,
                )
            return client.run(
                circuit,
                device=device,
                asynchronous=asynchronous,
                job_name=job_name,
                shots=shots,
                tags=tags,
            )

        return _submit()

    def _wait_with_retry(self, job_ids: Any, timeout: float | None = None) -> Any:
        """Wait for one or more jobs to finish, with retries."""
        client = self._require_client()

        @_retry_policy()
        def _wait() -> Any:
            return client.wait(job_ids, timeout=timeout)

        return _wait()

    def _estimate_with_retry(self, circuit: Any, device: str = "cpu") -> Any:
        """Estimate cost / runtime with retries."""
        client = self._require_client()

        @_retry_policy()
        def _estimate() -> Any:
            return client.estimate(circuit, device=device)

        return _estimate()

    @staticmethod
    def _job_result_to_dict(result: Any, include_counts: bool = True) -> dict[str, Any]:
        """Convert a BlueQubit ``JobResult`` into a plain dict."""
        # Runtime: prefer ``run_time_seconds`` (per spec), fall back to ms.
        runtime_seconds: float | None = None
        rt_s = getattr(result, "run_time_seconds", None)
        if rt_s is not None:
            runtime_seconds = float(rt_s)
        else:
            rt_ms = getattr(result, "run_time_ms", None)
            if rt_ms is not None:
                runtime_seconds = float(rt_ms) / 1000.0

        counts: dict[str, float] | None = None
        if include_counts:
            try:
                counts = result.get_counts()
            except BQError:
                counts = None
            except Exception:  # pragma: no cover - defensive
                counts = None

        return {
            "counts": counts,
            "job_id": getattr(result, "job_id", None),
            "cost": getattr(result, "cost", None),
            "runtime": runtime_seconds,
            "device": getattr(result, "device", None),
            "run_status": getattr(result, "run_status", None),
            "shots": getattr(result, "shots", None),
            "ok": getattr(result, "ok", None),
            "error_message": getattr(result, "error_message", None),
        }
