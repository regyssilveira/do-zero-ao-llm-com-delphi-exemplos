from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
EXPECTED_RANGES = {
    "corpus-mini.txt": (5 * 1024, 12 * 1024),
    "corpus-base.txt": (350 * 1024, 600 * 1024),
}


def fail(message: str) -> None:
    raise SystemExit(f"FALHOU: {message}")


def main() -> None:
    sums_path = DATASETS / "SHA256SUMS"
    declared = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        declared[name.strip()] = digest

    union: set[str] = set()
    for name, (minimum, maximum) in EXPECTED_RANGES.items():
        path = DATASETS / name
        raw = path.read_bytes()
        if not minimum <= len(raw) <= maximum:
            fail(f"{name} possui {len(raw)} bytes")
        if raw.startswith(b"\xef\xbb\xbf"):
            fail(f"{name} contém BOM")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            fail(f"{name} não é UTF-8: {error}")
        if text != unicodedata.normalize("NFC", text):
            fail(f"{name} não está normalizado em NFC")
        if "\r" in text:
            fail(f"{name} não usa apenas LF")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
            fail(f"{name} contém unidade substituta isolada")
        controls = {char for char in text if ord(char) < 32 and char not in "\n\t"}
        if controls:
            fail(f"{name} contém controles inesperados: {sorted(map(ord, controls))}")
        digest = hashlib.sha256(raw).hexdigest()
        if declared.get(name) != digest:
            fail(f"hash divergente para {name}")
        union.update(text)
        print(f"OK: {name}: {len(raw)} bytes; {len(set(text))} caracteres distintos")

    if len(union) > 128:
        fail(f"vocabulário conjunto excede 128 unidades: {len(union)}")
    print(f"OK: vocabulário conjunto: {len(union)} unidades UTF-16 no corpus auditado")


if __name__ == "__main__":
    main()
