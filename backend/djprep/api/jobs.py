"""In-process background jobs for analysis.

Analysis takes ~10 s for a 5-minute track, which is far too long to hold an HTTP
request open. A thread pool plus a progress-reporting job record is the right
size of solution for a single-user desktop-style tool; the interface
(`submit` -> job id -> poll) is the same one a Celery/RQ worker would present, so
swapping in a real queue later is a change of implementation, not of contract.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    stage: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "progress": round(self.progress, 3), "stage": self.stage,
                "error": self.error,
                "elapsed_sec": round((self.finished_at or time.time()) - self.created_at, 2)}


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Callable[[str, float], None]], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def progress(stage: str, frac: float) -> None:
            job.stage, job.progress = stage, float(frac)

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn(progress)
                job.status = "done"
                job.progress = 1.0
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                job.finished_at = time.time()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def prune(self, older_than: float = 3600.0) -> None:
        cut = time.time() - older_than
        with self._lock:
            for jid in [j for j, job in self._jobs.items()
                        if job.finished_at and job.finished_at < cut]:
                self._jobs.pop(jid, None)
