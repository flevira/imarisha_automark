from services import extraction_service


def test_run_extraction_uses_prepared_input_and_restores_source(monkeypatch) -> None:
    cleanup_calls: list[str | None] = []

    class Prepared:
        path = "/tmp/input_ocr.pdf"
        cleanup_path = "/tmp/cleanup_dir"
        enhanced = True

    def fake_prepare_extraction_input(file_path: str):
        assert file_path == "/data/original_scan.pdf"
        return Prepared()

    def fake_extract_file(file_path: str, student_id_mode: str):
        assert file_path == "/tmp/input_ocr.pdf"
        assert student_id_mode == "auto"
        return [{"source_file": file_path, "needs_review": False}]

    def fake_cleanup(path: str | None):
        cleanup_calls.append(path)

    monkeypatch.setattr(extraction_service, "prepare_extraction_input", fake_prepare_extraction_input)
    monkeypatch.setattr(extraction_service, "extract_file", fake_extract_file)
    monkeypatch.setattr(extraction_service, "cleanup_prepared_input", fake_cleanup)

    rows = extraction_service.run_extraction(["/data/original_scan.pdf"], "auto")

    assert rows[0]["source_file"] == "/data/original_scan.pdf"
    assert rows[0]["ocr_enhanced_input"] is True
    assert cleanup_calls == ["/tmp/cleanup_dir"]


def test_run_extraction_without_enhancement_marks_flag(monkeypatch) -> None:
    cleanup_calls: list[str | None] = []

    class Prepared:
        path = "/data/original_text_layer.pdf"
        cleanup_path = None
        enhanced = False

    monkeypatch.setattr(
        extraction_service,
        "prepare_extraction_input",
        lambda file_path: Prepared(),
    )
    monkeypatch.setattr(
        extraction_service,
        "extract_file",
        lambda file_path, student_id_mode: [{"source_file": file_path}],
    )
    monkeypatch.setattr(
        extraction_service,
        "cleanup_prepared_input",
        lambda path: cleanup_calls.append(path),
    )

    rows = extraction_service.run_extraction(["/data/original_text_layer.pdf"], "qr")

    assert rows[0]["source_file"] == "/data/original_text_layer.pdf"
    assert rows[0]["ocr_enhanced_input"] is False
    assert cleanup_calls == [None]
