from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .asset_paths import to_relative_project_path
from .ohr_inventory import resolve_ohr_inventory_path
from .pdf_preprocessing import export_docling_markdown
from .resource_limits import enforce_memory_budget


STAGE_ORDER = ("extract_pdf", "docling_audit", "render_pages", "got_ocr")


def page_image_name(doc_rel: str, page_id: int) -> str:
    return f"{doc_rel}_page_{page_id}.png"


def page_ocr_name(doc_rel: str, page_id: int) -> str:
    return f"{doc_rel}_page_{page_id}.txt"


def docling_audit_name(doc_rel: str) -> str:
    return f"{doc_rel}.docling.md"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_got_ocr(project_root: Path) -> dict[str, str]:
    payload = json.loads((project_root / "config/model_revisions.json").read_text(encoding="utf-8"))
    locked = payload["models"]["got_ocr"]
    repository = str(locked["repository"]).strip()
    revision = str(locked["revision"]).strip()
    if not repository or not revision:
        raise ValueError("config/model_revisions.json got_ocr lock is incomplete.")
    return {"repository": repository, "revision": revision}


def _rel(path: Path, project_root: Path) -> str:
    try:
        return to_relative_project_path(path, project_root)
    except Exception:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _remove_quietly(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        if os.uname().sysname == "Darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def plan_document_work(
    *,
    project_root: Path,
    doc_rel: str,
    pdf_path: Path,
    page_ids: list[int],
    out_root: Path,
    got_ocr: dict[str, str] | None = None,
) -> dict[str, Any]:
    image_dir = out_root / "images"
    ocr_dir = out_root / "ocr"
    audit_dir = out_root / "docling"
    audit_output = audit_dir / docling_audit_name(doc_rel)
    image_outputs = [image_dir / page_image_name(doc_rel, page_id) for page_id in page_ids]
    ocr_outputs = [ocr_dir / page_ocr_name(doc_rel, page_id) for page_id in page_ids]
    stages = {
        "extract_pdf": {
            "output": _rel(pdf_path, project_root),
            "done": pdf_path.is_file(),
        },
        "docling_audit": {
            "input": _rel(pdf_path, project_root),
            "output": _rel(audit_output, project_root),
            "done": audit_output.is_file() and audit_output.stat().st_size > 0,
        },
        "render_pages": {
            "outputs": [_rel(path, project_root) for path in image_outputs],
            "done": all(path.is_file() and path.stat().st_size > 0 for path in image_outputs),
        },
        "got_ocr": {
            "outputs": [_rel(path, project_root) for path in ocr_outputs],
            "done": all(path.is_file() and path.stat().st_size > 0 for path in ocr_outputs),
            "model": got_ocr or {},
        },
    }
    return {
        "doc_name": doc_rel,
        "pdf": _rel(pdf_path, project_root),
        "page_ids": page_ids,
        "stages": stages,
        "checkpoint": {stage: stages[stage]["done"] for stage in STAGE_ORDER},
    }


def resolve_pdf_source(
    *,
    project_root: Path,
    doc_rel: str,
    pdf_root: Path | None,
    pdf_zip: Path | None,
    inventory_dir: Path,
) -> tuple[str, Path | None]:
    """Return (member_or_path_kind, source_path_or_zip_member_path)."""
    _path, resolved_key, _kind = resolve_ohr_inventory_path(inventory_dir, doc_rel)
    keys = [doc_rel]
    if resolved_key and resolved_key not in keys:
        keys.append(resolved_key)

    if pdf_root is not None:
        root = pdf_root if pdf_root.is_absolute() else project_root / pdf_root
        for key in keys:
            candidate = root / f"{key}.pdf"
            if candidate.is_file():
                return "filesystem", candidate

    zip_path = pdf_zip or (project_root / "data/ohr_bench_raw/pdfs.zip")
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        for key in keys:
            member = f"{key}.pdf"
            if member in names:
                return "zip", Path(member)
    return "missing", None


def extract_pdf_from_zip(zip_path: Path, member: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def render_pdf_pages(pdf_path: Path, page_ids: list[int], image_paths: dict[int, Path], scale: float = 2.0) -> dict[str, Any]:
    import pypdfium2 as pdfium

    started = time.perf_counter()
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        pdf_page_count = len(document)
        for page_id in page_ids:
            if page_id < 0 or page_id >= pdf_page_count:
                raise ValueError(
                    f"Inventory page_id {page_id} is outside PDF page range 0..{pdf_page_count - 1} for {pdf_path}"
                )
            output = image_paths[page_id]
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_suffix(output.suffix + ".partial")
            _remove_quietly(partial)
            page = document[page_id]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            image.save(partial, format="PNG")
            if partial.stat().st_size <= 0:
                _remove_quietly(partial)
                raise RuntimeError(f"Rendered empty PNG for page {page_id}")
            partial.replace(output)
    finally:
        document.close()
    return {
        "pdf_page_count": pdf_page_count,
        "render_runtime_sec": time.perf_counter() - started,
        "render_scale": scale,
    }


@dataclass
class PreparationResult:
    doc_name: str
    page_ids: list[int]
    pdf_path: Path
    pdf_sha256: str
    got_ocr_repository: str
    got_ocr_revision: str
    device: str
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


def execute_document_preparation(
    *,
    project_root: Path,
    doc_rel: str,
    page_ids: list[int],
    out_root: Path,
    pdf_root: Path | None = None,
    pdf_zip: Path | None = None,
    inventory_dir: Path | None = None,
    extract_got_ocr_fn: Callable[..., str] | None = None,
    export_docling_fn: Callable[[Path, Path], Path] = export_docling_markdown,
) -> PreparationResult:
    project_root = project_root.expanduser().resolve()
    out_root = out_root if out_root.is_absolute() else project_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    inventory_dir = (inventory_dir or project_root / "OHR-Bench/data/retrieval_base/gt").resolve()
    locked = load_locked_got_ocr(project_root)
    device = _device_name()
    peak_rss = _rss_bytes()

    kind, source = resolve_pdf_source(
        project_root=project_root,
        doc_rel=doc_rel,
        pdf_root=pdf_root,
        pdf_zip=pdf_zip,
        inventory_dir=inventory_dir,
    )
    if kind == "missing" or source is None:
        raise FileNotFoundError(f"Could not locate PDF for document {doc_rel!r}.")

    pdf_path = out_root / "pdfs" / f"{doc_rel}.pdf"
    provenance_path = out_root / "provenance" / f"{doc_rel}.json"
    image_dir = out_root / "images"
    ocr_dir = out_root / "ocr"
    audit_path = out_root / "docling" / docling_audit_name(doc_rel)
    image_paths = {page_id: image_dir / page_image_name(doc_rel, page_id) for page_id in page_ids}
    ocr_paths = {page_id: ocr_dir / page_ocr_name(doc_rel, page_id) for page_id in page_ids}

    metrics: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "device": device,
        "page_ids": page_ids,
        "got_ocr_repository": locked["repository"],
        "got_ocr_revision": locked["revision"],
        "per_page_got_ocr_runtime_sec": {},
        "skipped_pages": {"render": [], "ocr": []},
    }

    try:
        # Stage: extract PDF
        enforce_memory_budget("asset preparation before PDF extraction")
        if kind == "filesystem":
            assert isinstance(source, Path)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            if not pdf_path.exists() or sha256_file(pdf_path) != sha256_file(source):
                shutil.copy2(source, pdf_path)
        else:
            zip_path = (pdf_zip or project_root / "data/ohr_bench_raw/pdfs.zip").resolve()
            member = source.as_posix()
            extract_pdf_from_zip(zip_path, member, pdf_path)
        pdf_sha256 = sha256_file(pdf_path)
        metrics["pdf_sha256"] = pdf_sha256
        metrics["pdf_bytes"] = pdf_path.stat().st_size
        enforce_memory_budget("asset preparation after PDF extraction")

        existing = _read_json(provenance_path) or {}
        # Stage: Docling audit
        enforce_memory_budget("asset preparation before Docling audit")
        docling_started = time.perf_counter()
        expected_audit = {
            "pdf_sha256": pdf_sha256,
            "stage": "docling_audit",
        }
        audit_prov = (existing.get("docling_audit") or {}) if isinstance(existing, dict) else {}
        if (
            audit_path.is_file()
            and audit_path.stat().st_size > 0
            and audit_prov.get("pdf_sha256") == pdf_sha256
        ):
            metrics["docling_runtime_sec"] = 0.0
            metrics["docling_skipped"] = True
        else:
            partial = audit_path.with_suffix(audit_path.suffix + ".partial")
            _remove_quietly(partial)
            export_docling_fn(pdf_path, partial)
            if not partial.is_file() or partial.stat().st_size <= 0:
                _remove_quietly(partial)
                raise RuntimeError("Docling produced empty audit output.")
            partial.replace(audit_path)
            metrics["docling_runtime_sec"] = time.perf_counter() - docling_started
            metrics["docling_skipped"] = False
            existing["docling_audit"] = expected_audit
        enforce_memory_budget("asset preparation after Docling audit")

        # Stage: render pages
        enforce_memory_budget("asset preparation before page rendering")
        pages_to_render: list[int] = []
        for page_id in page_ids:
            image_path = image_paths[page_id]
            page_prov = ((existing.get("pages") or {}).get(str(page_id)) or {})
            if (
                image_path.is_file()
                and image_path.stat().st_size > 0
                and page_prov.get("pdf_sha256") == pdf_sha256
                and page_prov.get("png_sha256") == sha256_file(image_path)
            ):
                metrics["skipped_pages"]["render"].append(page_id)
            else:
                pages_to_render.append(page_id)
                _remove_quietly(image_path)
                _remove_quietly(ocr_paths[page_id])  # OCR depends on image contents.
        render_meta = {"pdf_page_count": None, "render_runtime_sec": 0.0, "render_scale": 2.0}
        if pages_to_render:
            render_meta = render_pdf_pages(
                pdf_path,
                pages_to_render,
                {page_id: image_paths[page_id] for page_id in pages_to_render},
            )
        # Validate all inventory pages exist after render/skip.
        for page_id in page_ids:
            image_path = image_paths[page_id]
            if not image_path.is_file() or image_path.stat().st_size <= 0:
                raise RuntimeError(f"Missing rendered PNG for page {page_id}")
        metrics["pdf_page_count"] = render_meta.get("pdf_page_count")
        if metrics["pdf_page_count"] is None:
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(str(pdf_path))
            try:
                metrics["pdf_page_count"] = len(document)
            finally:
                document.close()
        expected_pages = set(range(int(metrics["pdf_page_count"])))
        if set(page_ids) != expected_pages:
            raise RuntimeError(
                f"Inventory page ids {page_ids} do not match PDF pages {sorted(expected_pages)}."
            )
        metrics["render_runtime_sec"] = float(render_meta.get("render_runtime_sec") or 0.0)
        metrics["render_scale"] = render_meta.get("render_scale", 2.0)
        enforce_memory_budget("asset preparation after page rendering")

        # Stage: GOT-OCR — repository/revision come only from committed lock file.
        model_name = locked["repository"]
        revision = locked["revision"]
        ocr_total = 0.0
        pages_state = dict(existing.get("pages") or {})
        ocr_fn = extract_got_ocr_fn
        for page_id in page_ids:
            enforce_memory_budget(f"asset preparation before GOT-OCR page {page_id}")
            image_path = image_paths[page_id]
            ocr_path = ocr_paths[page_id]
            png_sha = sha256_file(image_path)
            page_prov = pages_state.get(str(page_id)) or {}
            if (
                ocr_path.is_file()
                and ocr_path.stat().st_size > 0
                and page_prov.get("pdf_sha256") == pdf_sha256
                and page_prov.get("png_sha256") == png_sha
                and page_prov.get("got_ocr_revision") == revision
                and page_prov.get("ocr_sha256") == sha256_file(ocr_path)
            ):
                metrics["skipped_pages"]["ocr"].append(page_id)
                enforce_memory_budget(f"asset preparation after GOT-OCR page {page_id}")
                continue
            # Lazy-import Torch/Transformers OCR only when a page actually needs OCR.
            if ocr_fn is None:
                from .ocr import extract_got_ocr

                ocr_fn = extract_got_ocr
            partial = ocr_path.with_suffix(ocr_path.suffix + ".partial")
            _remove_quietly(partial)
            _remove_quietly(ocr_path)
            started = time.perf_counter()
            text = ocr_fn(image_path, model_name=model_name, revision=revision)
            runtime = time.perf_counter() - started
            ocr_total += runtime
            metrics["per_page_got_ocr_runtime_sec"][str(page_id)] = runtime
            if not str(text).strip():
                raise RuntimeError(f"GOT-OCR returned empty text for page {page_id}")
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(str(text).rstrip() + "\n", encoding="utf-8")
            if partial.stat().st_size <= 0:
                _remove_quietly(partial)
                raise RuntimeError(f"Wrote empty OCR text for page {page_id}")
            partial.replace(ocr_path)
            pages_state[str(page_id)] = {
                "pdf_sha256": pdf_sha256,
                "png_sha256": png_sha,
                "ocr_sha256": sha256_file(ocr_path),
                "got_ocr_repository": model_name,
                "got_ocr_revision": revision,
                "image_path": _rel(image_path, project_root),
                "ocr_text_path": _rel(ocr_path, project_root),
                "bytes_png": image_path.stat().st_size,
                "bytes_ocr": ocr_path.stat().st_size,
            }
            current_rss = _rss_bytes()
            if current_rss is not None:
                peak_rss = max(peak_rss or 0, current_rss)
            enforce_memory_budget(f"asset preparation after GOT-OCR page {page_id}")

        metrics["got_ocr_runtime_sec_total"] = ocr_total
        metrics["peak_rss_bytes"] = peak_rss
        metrics["finished_at_utc"] = datetime.now(UTC).isoformat()

        storage = {
            "png_bytes_total": sum(image_paths[page_id].stat().st_size for page_id in page_ids),
            "ocr_bytes_total": sum(ocr_paths[page_id].stat().st_size for page_id in page_ids),
            "per_page_bytes": {
                str(page_id): {
                    "png": image_paths[page_id].stat().st_size,
                    "ocr": ocr_paths[page_id].stat().st_size,
                }
                for page_id in page_ids
            },
        }
        metrics["storage"] = storage

        provenance = {
            "doc_name": doc_rel,
            "pdf_sha256": pdf_sha256,
            "pdf_path": _rel(pdf_path, project_root),
            "page_ids": page_ids,
            "got_ocr_repository": model_name,
            "got_ocr_revision": revision,
            "device": device,
            "docling_audit": {
                "pdf_sha256": pdf_sha256,
                "output": _rel(audit_path, project_root),
            },
            "pages": pages_state,
            "metrics": {
                "render_runtime_sec": metrics["render_runtime_sec"],
                "got_ocr_runtime_sec_total": metrics["got_ocr_runtime_sec_total"],
                "per_page_got_ocr_runtime_sec": metrics["per_page_got_ocr_runtime_sec"],
                "peak_rss_bytes": metrics["peak_rss_bytes"],
            },
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(provenance_path, provenance)

        return PreparationResult(
            doc_name=doc_rel,
            page_ids=page_ids,
            pdf_path=pdf_path,
            pdf_sha256=pdf_sha256,
            got_ocr_repository=model_name,
            got_ocr_revision=revision,
            device=device,
            metrics=metrics,
            outputs={
                "pdf": _rel(pdf_path, project_root),
                "docling_audit": _rel(audit_path, project_root),
                "images": [_rel(image_paths[page_id], project_root) for page_id in page_ids],
                "ocr": [_rel(ocr_paths[page_id], project_root) for page_id in page_ids],
                "provenance": _rel(provenance_path, project_root),
            },
        )
    except Exception:
        # Never silently keep partial outputs from a failed attempt.
        for path in list(image_paths.values()) + list(ocr_paths.values()) + [audit_path]:
            partial = path.with_suffix(path.suffix + ".partial")
            _remove_quietly(partial)
        # Remove zero-byte outputs created during the failed run.
        for path in list(image_paths.values()) + list(ocr_paths.values()) + [audit_path]:
            if path.is_file() and path.stat().st_size <= 0:
                _remove_quietly(path)
        raise
