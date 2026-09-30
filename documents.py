from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 300


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    number: int
    page: int | None = None


def split_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Overlap, sifir ile parca boyutu arasinda olmali.")
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max(size // 2, overlap + 1), end)
            if boundary != -1:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def load_documents(folder: Path) -> tuple[list[Chunk], list[str]]:
    if not folder.is_dir():
        raise ValueError(f"Belge klasoru bulunamadi: {folder}")

    chunks = []
    warnings = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        file_chunks = []
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Dosya 10 MB sinirini asiyor.")
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(BytesIO(path.read_bytes()))
                if reader.is_encrypted:
                    raise ValueError("Sifreli PDF desteklenmiyor.")
                if len(reader.pages) > 100:
                    raise ValueError("PDF 100 sayfa sinirini asiyor.")
                pages = (
                    (i, page.extract_text() or "")
                    for i, page in enumerate(reader.pages, 1)
                )
            else:
                pages = [(None, path.read_text(encoding="utf-8-sig"))]

            blank_pages = 0
            for page, text in pages:
                parts = split_text(text)
                if not parts:
                    blank_pages += 1
                for part in parts:
                    file_chunks.append(
                        Chunk(part, path.name, len(file_chunks) + 1, page)
                    )
                    if len(file_chunks) > MAX_CHUNKS:
                        raise ValueError(f"Dosya {MAX_CHUNKS} parca sinirini asiyor.")
            if not file_chunks:
                raise ValueError("Okunabilir metin yok. Taranmis PDF icin OCR gerekir.")
            if blank_pages:
                warnings.append(
                    f"{path.name}: {blank_pages} bos/metinsiz sayfa atlandi."
                )
        except (OSError, UnicodeError, ValueError, PyPdfError) as exc:
            warnings.append(f"{path.name} atlandi: {exc}")
            continue

        if len(chunks) + len(file_chunks) > MAX_CHUNKS:
            raise ValueError(
                f"Toplam {MAX_CHUNKS} parca siniri asildi. Daha az belge kullanin."
            )
        chunks.extend(file_chunks)
    return chunks, warnings
