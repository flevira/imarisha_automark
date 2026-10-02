"""Keep the legacy public entry point without loading OCR for geometry-only work."""

__all__ = ["extract_file"]


def __getattr__(name):
    if name == "extract_file":
        from core.extractor import extract_file
        return extract_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
