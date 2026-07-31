"""
backend/agents/doc_parser_agent/guardrails.py
=============================================
File validation guardrails for the DocParser agent.

Two guardrails, checked in order:
    1. FileSizeGuardrail  — rejects files > MAX_FILE_SIZE_MB before reading content
    2. FileTypeGuardrail  — validates file type using magic bytes (not extension)

Magic bytes rationale:
    File extensions can be trivially spoofed (rename .exe to .pdf).
    Magic bytes are the actual binary signature at the start of the file.
    - PDF:  starts with b'%PDF-'  (bytes: 25 50 44 46 2D)
    - DOCX: starts with b'PK\x03\x04' (ZIP archive — all OOXML formats are ZIP)

Both implement the GuardrailStrategy protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.base.guardrail import GuardrailStrategy
from backend.state import GuardrailResult

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Magic bytes signatures
PDF_MAGIC  = b"%PDF-"
DOCX_MAGIC = b"PK\x03\x04"   # ZIP PK header — covers all OOXML (.docx, .xlsx, .pptx)

MAGIC_READ_BYTES = 8          # Only need first 8 bytes for classification


# ── FileSizeGuardrail ──────────────────────────────────────────────────────────

class FileSizeGuardrail(GuardrailStrategy):
    """
    Reject files larger than MAX_FILE_SIZE_MB before reading any content.

    Reads only the file size via os.stat — no file content is loaded.
    This prevents memory exhaustion from oversized uploads before any
    processing begins.
    """

    def __init__(self, max_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
        self._max_bytes = max_bytes

    async def check(self, query: str) -> GuardrailResult:
        """
        Check file size.

        Parameters
        ----------
        query : str
            Absolute path to the uploaded file.
        """
        file_path = Path(query)

        if not file_path.exists():
            return GuardrailResult.fail(
                reason=f"File not found: {file_path.name}",
                guardrail_name="FileSizeGuardrail",
            )

        size_bytes = file_path.stat().st_size

        if size_bytes > self._max_bytes:
            size_mb = size_bytes / (1024 * 1024)
            logger.warning(
                "FileSizeGuardrail: rejected %s (%.1f MB > %d MB limit)",
                file_path.name, size_mb, MAX_FILE_SIZE_MB,
            )
            return GuardrailResult.fail(
                reason=(
                    f"File '{file_path.name}' is {size_mb:.1f} MB, "
                    f"which exceeds the {MAX_FILE_SIZE_MB} MB limit. "
                    "Please upload a smaller file."
                ),
                guardrail_name="FileSizeGuardrail",
            )

        return GuardrailResult.ok()


# ── FileTypeGuardrail ──────────────────────────────────────────────────────────

class FileTypeGuardrail(GuardrailStrategy):
    """
    Validate file type using magic bytes — not file extension.

    Accepts:
        - PDF   (magic: %PDF-)
        - DOCX  (magic: PK\x03\x04 — ZIP container for OOXML)

    Rejects everything else with a clear user message.
    """

    async def check(self, query: str) -> GuardrailResult:
        """
        Check file type via magic bytes.

        Parameters
        ----------
        query : str
            Absolute path to the uploaded file.
        """
        file_path = Path(query)

        try:
            with file_path.open("rb") as f:
                header = f.read(MAGIC_READ_BYTES)
        except OSError as exc:
            return GuardrailResult.fail(
                reason=f"Could not read file '{file_path.name}': {exc}",
                guardrail_name="FileTypeGuardrail",
            )

        if header.startswith(PDF_MAGIC):
            logger.debug("FileTypeGuardrail: %s identified as PDF", file_path.name)
            return GuardrailResult.ok()

        if header.startswith(DOCX_MAGIC):
            logger.debug("FileTypeGuardrail: %s identified as DOCX (ZIP/OOXML)", file_path.name)
            return GuardrailResult.ok()

        # Unrecognised magic bytes
        logger.warning(
            "FileTypeGuardrail: rejected %s (magic bytes: %s)",
            file_path.name, header.hex(),
        )
        return GuardrailResult.fail(
            reason=(
                f"'{file_path.name}' is not a supported file type. "
                "Please upload a PDF (.pdf) or Word document (.docx)."
            ),
            guardrail_name="FileTypeGuardrail",
        )
