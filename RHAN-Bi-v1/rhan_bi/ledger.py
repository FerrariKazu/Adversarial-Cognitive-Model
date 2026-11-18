"""Append-only, content-addressed evidence ledger (docs/03)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from rhan_bi.types import EvidenceRecord


class LedgerIntegrityError(RuntimeError):
    pass


class EvidenceLedger:
    def __init__(self, max_images: int = 256):
        self._records: list[EvidenceRecord] = []
        self._by_id: dict[str, EvidenceRecord] = {}
        self._images: dict[str, np.ndarray] = {}
        self._image_order: list[str] = []
        self._max_images = int(max_images)

    # ------------------------------------------------------------------ write
    def add(self, record: EvidenceRecord, image: np.ndarray | None = None) -> EvidenceRecord:
        if record.record_id in self._by_id:
            raise LedgerIntegrityError(f"duplicate record id {record.record_id}")
        if not record.verify():
            raise LedgerIntegrityError("record id does not match its payload")
        for rel in record.dependence.get("relations", []):
            if rel["record_id"] not in self._by_id:
                raise LedgerIntegrityError(f"dependence refers to unknown record {rel['record_id']}")
        self._records.append(record)
        self._by_id[record.record_id] = record
        if image is not None:
            self._remember_image(record.image_sha256, image)
        return record

    def _remember_image(self, key: str, image: np.ndarray) -> None:
        if key in self._images:
            return
        self._images[key] = np.asarray(image)
        self._image_order.append(key)
        while len(self._image_order) > self._max_images:
            old = self._image_order.pop(0)
            self._images.pop(old, None)

    # ------------------------------------------------------------------ read
    @property
    def records(self) -> list[EvidenceRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[EvidenceRecord]:
        return iter(self._records)

    def get(self, record_id: str) -> EvidenceRecord:
        return self._by_id[record_id]

    def image(self, image_sha256: str) -> np.ndarray | None:
        return self._images.get(image_sha256)

    def last(self, n: int = 1) -> list[EvidenceRecord]:
        return self._records[-n:] if n else []

    def find(self, **filters) -> list[EvidenceRecord]:
        out = []
        for rec in self._records:
            ok = True
            for key, value in filters.items():
                if key == "view":
                    ok &= rec.action.get("view") == value
                elif key == "light":
                    ok &= rec.action.get("light") == value
                elif key == "kind":
                    ok &= rec.action.get("kind") == value
                else:
                    ok &= getattr(rec, key) == value
            if ok:
                out.append(rec)
        return out

    def total_rho(self) -> float:
        return float(sum(r["rho"] for rec in self._records for r in rec.dependence.get("relations", [])))

    def n_eff(self) -> float:
        rels = [r["rho"] for rec in self._records for r in rec.dependence.get("relations", [])]
        if not rels:
            return float(len(self._records))
        avg = float(np.mean(rels))
        n = len(self._records)
        return float(n / (1.0 + (n - 1) * max(0.0, avg)))

    # ------------------------------------------------------------------ integrity
    def verify(self) -> dict[str, object]:
        for rec in self._records:
            if not rec.verify():
                raise LedgerIntegrityError(f"record {rec.record_id} fails its hash check")
            if "cause" in rec.to_json():
                raise LedgerIntegrityError("ground truth leaked into a record")
        for i, rec in enumerate(self._records):
            for rel in rec.dependence.get("relations", []):
                ids = [r.record_id for r in self._records[:i]]
                if rel["record_id"] not in ids:
                    raise LedgerIntegrityError("dependence relation is not to an earlier record")
        return {"records": len(self._records), "ok": True}

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_json(), sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, verify: bool = True) -> "EvidenceLedger":
        ledger = cls()
        with Path(path).open() as fh:
            for line in fh:
                if line.strip():
                    ledger.add(EvidenceRecord.from_json(json.loads(line)))
        if verify:
            ledger.verify()
        return ledger
