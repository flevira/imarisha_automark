
import flet as ft
import json
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

from services.extraction_service import run_extraction
from app.views.review_view import build_review_view

ENABLE_TEMP_CSV_EXPORT = True


def build_home_view(page: ft.Page) -> None:
    page.controls.clear()

    selected_file_paths: list[str] = []
    latest_results: list[dict] = []
    selected_files = ft.Text("No file selected")
    extraction_stats = ft.Text("", selectable=True, visible=False)
    mode_dropdown = ft.Dropdown(
        label="Student ID source",
        value="auto",
        options=[
            ft.dropdown.Option("qr"),
            ft.dropdown.Option("text"),
            ft.dropdown.Option("auto"),
        ],
        width=220,
    )

    def on_pick_result(e: ft.FilePickerResultEvent) -> None:
        print("[home_view] FilePicker result received")
        if e.files:
            print(f"[home_view] Files selected count: {len(e.files)}")
            selected_file_paths.clear()
            selected_names: list[str] = []
            for file in e.files:
                selected_names.append(file.name)
                if file.path:
                    selected_file_paths.append(file.path)

            if selected_file_paths:
                selected_files.value = "\n".join(selected_names)
            else:
                selected_files.value = (
                    "⚠️ Files were selected, but local file paths were not provided.\n"
                    "Run the app with desktop mode (flet-desktop) so extraction can read local files."
                )
        else:
            selected_files.value = "No file selected"

        page.update()

    picker = ft.FilePicker(on_result=on_pick_result)
    page.overlay.append(picker)
    page.file_picker = picker
    page.update()

    def _set_selected_paths(paths: list[str]) -> None:
        selected_file_paths.clear()
        selected_file_paths.extend(paths)
        selected_files.value = "\n".join([Path(path).name for path in paths]) if paths else "No file selected"
        page.update()

    def _open_native_file_dialog() -> list[str]:
        try:
            # On macOS, using a Tk child process can create a separate Python app icon
            # and unstable folder navigation behavior. Use AppleScript for a native
            # open panel that stays in front without spawning a Python GUI app.
            if sys.platform == "darwin":
                script = """
set selectedFiles to choose file with prompt "Choose answer sheet files" of type {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"} with multiple selections allowed
set outputLines to {}
repeat with selectedFile in selectedFiles
    set end of outputLines to POSIX path of selectedFile
end repeat
set AppleScript's text item delimiters to linefeed
return outputLines as text
""".strip()

                proc = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                # User cancellation on osascript returns non-zero and "User canceled."
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip().lower()
                    if "user canceled" in stderr:
                        _set_selected_paths([])
                        return
                    raise RuntimeError(proc.stderr.strip() or "AppleScript file dialog failed.")

                selected = [
                    line.strip()
                    for line in proc.stdout.splitlines()
                    if line.strip()
                ]
                return [str(Path(path).expanduser().resolve()) for path in selected]

            # Tkinter dialogs on macOS must run on the main thread.
            # Flet click handlers can execute outside the main thread, so we
            # launch a short-lived child process to own the native dialog.
            helper_code = """
import json
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.update_idletasks()
root.lift()
root.attributes("-topmost", True)
root.update()
selected = list(
    filedialog.askopenfilenames(
        title="Choose answer sheet files",
        parent=root,
        filetypes=[
            ("Supported files", "*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
)
root.attributes("-topmost", False)
root.destroy()
print(json.dumps(selected))
""".strip()

            proc = subprocess.run(
                [sys.executable, "-c", helper_code],
                capture_output=True,
                text=True,
                check=True,
            )
            selected = json.loads(proc.stdout.strip() or "[]")
            return [str(Path(path).expanduser().resolve()) for path in selected]
        except Exception as exc:
            raise RuntimeError(f"Native file dialog failed: {exc}") from exc

    def select_files_clicked(_: ft.ControlEvent) -> None:
        print("[home_view] Select file(s) button clicked")
        try:
            # FilePicker is preferred in Flet and works reliably on most platforms.
            # On macOS desktop builds, the native picker can become unresponsive in
            # some setups, so we use the native OS dialog directly.
            if sys.platform == "darwin":
                _set_selected_paths(_open_native_file_dialog())
                return

            print("[home_view] Calling FilePicker.pick_files()")
            picker.pick_files(
                dialog_title="Choose answer sheet files",
                allow_multiple=True,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            )
        except Exception:
            # Single CTA behavior: transparently fall back to a native dialog if
            # Flet FilePicker is unavailable in this runtime.
            try:
                _set_selected_paths(_open_native_file_dialog())
            except Exception as exc:  # pragma: no cover - UI fallback
                selected_files.value = f"⚠️ Could not open file browser: {exc}"
                page.update()

    def run_clicked(_: ft.ControlEvent) -> None:
        nonlocal latest_results
        if not selected_file_paths:
            selected_files.value = "⚠️ No file selected"
            page.update()
            return

        results = run_extraction(
            file_paths=selected_file_paths,
            student_id_mode=mode_dropdown.value or "auto",
        )
        latest_results = results

        status_message = f"✅ Extracted {len(results)} rows"
        if ENABLE_TEMP_CSV_EXPORT and results:
            try:
                first_file = Path(selected_file_paths[0]).expanduser().resolve()
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                export_path = first_file.parent / f"extracted_preview_{timestamp}.csv"

                fieldnames: list[str] = []
                for row in results:
                    for key in row.keys():
                        if key not in fieldnames:
                            fieldnames.append(key)

                with export_path.open("w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(results)

                status_message += f"\n📝 Temp CSV preview: {export_path}"
            except Exception as exc:  # pragma: no cover - temporary UI safety
                status_message += f"\n⚠️ Could not save temp CSV preview: {exc}"

        selected_files.value = status_message
        needs_review = sum(1 for row in results if row.get("needs_review"))
        manual_marking = sum(1 for row in results if row.get("manual_marking"))
        unique_students = len({row.get("student_id") for row in results if row.get("student_id")})
        sections: dict[str, int] = {}
        for row in results:
            section_name = str(row.get("section") or "UNKNOWN")
            sections[section_name] = sections.get(section_name, 0) + 1

        page_quality: dict[tuple[str, int], dict] = {}
        for row in results:
            source_file = str(row.get("source_file") or "")
            page_number_raw = row.get("page")
            if not isinstance(page_number_raw, int):
                continue
            page_key = (source_file, page_number_raw)
            if page_key not in page_quality:
                page_quality[page_key] = row

        quality_scores = [
            float(page.get("scan_quality_score"))
            for page in page_quality.values()
            if isinstance(page.get("scan_quality_score"), (int, float))
        ]
        poor_pages = sum(
            1 for page in page_quality.values() if str(page.get("scan_quality_band")) == "poor"
        )
        fair_pages = sum(
            1 for page in page_quality.values() if str(page.get("scan_quality_band")) == "fair"
        )
        good_pages = sum(
            1 for page in page_quality.values() if str(page.get("scan_quality_band")) == "good"
        )
        avg_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else None

        section_lines = [f"- {name}: {count}" for name, count in sorted(sections.items())]
        quality_lines = (
            [
                f"- Pages analyzed for quality: {len(page_quality)}",
                f"- Average scan quality score: {avg_quality}/100",
                f"- Good scans: {good_pages}",
                f"- Fair scans: {fair_pages}",
                f"- Poor scans: {poor_pages}",
            ]
            if page_quality
            else ["- Scan quality unavailable"]
        )
        individual_quality_lines: list[str] = []
        for (source_file, page_number), page_data in sorted(
            page_quality.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            file_name = Path(source_file).name if source_file else "unknown-file"
            brightness = page_data.get("scan_brightness_mean", "n/a")
            contrast = page_data.get("scan_contrast_std", "n/a")
            blur_proxy = page_data.get("scan_blur_laplacian_var", "n/a")
            score = page_data.get("scan_quality_score", "n/a")
            band = page_data.get("scan_quality_band", "n/a")
            skew = page_data.get("scan_estimated_skew_deg", "n/a")
            individual_quality_lines.append(
                (
                    f"- {file_name} page {page_number}: "
                    f"brightness={brightness}, contrast={contrast}, "
                    f"blur_proxy={blur_proxy}, skew={skew}°, "
                    f"quality_score={score}/100, quality_band={band}"
                )
            )

        extraction_stats.value = "\n".join(
            [
                "Review statistics",
                f"- Total extracted rows: {len(results)}",
                f"- Rows flagged for review: {needs_review}",
                f"- Rows requiring manual marking: {manual_marking}",
                f"- Unique students: {unique_students}",
                "- Scan quality:",
                *quality_lines,
                "- Individual scan quality details:",
                *(individual_quality_lines if individual_quality_lines else ["- (none)"]),
                "- Section counts:",
                *(section_lines if section_lines else ["- (none)"]),
            ]
        )
        extraction_stats.visible = True
        print(results[:5])  # debug
        page.update()

    def open_review_clicked(_: ft.ControlEvent) -> None:
        build_review_view(page, extracted_rows=latest_results)

    page.add(
        ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Imarisha Automark", size=30, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Upload answer sheets, choose a Student ID source, and run extraction.",
                        size=14,
                        color=ft.Colors.BLUE_GREY_700,
                    ),
                    ft.Container(height=8),
                    ft.ResponsiveRow(
                        controls=[
                            ft.Container(
                                col={"sm": 12, "md": 4},
                                content=ft.ElevatedButton(
                                    "Select file(s)",
                                    icon=ft.Icons.UPLOAD_FILE,
                                    on_click=select_files_clicked,
                                    width=260,
                                    height=44,
                                ),
                            ),
                            ft.Container(col={"sm": 12, "md": 3}, content=mode_dropdown),
                            ft.Container(
                                col={"sm": 12, "md": 3},
                                content=ft.ElevatedButton(
                                    "Run extraction",
                                    icon=ft.Icons.PLAY_ARROW,
                                    on_click=run_clicked,
                                    width=220,
                                    height=44,
                                ),
                            ),
                            ft.Container(
                                col={"sm": 12, "md": 2},
                                content=ft.OutlinedButton(
                                    "Open review",
                                    icon=ft.Icons.RATE_REVIEW,
                                    on_click=open_review_clicked,
                                    width=180,
                                    height=44,
                                ),
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        run_spacing=12,
                        spacing=12,
                    ),
                    ft.Container(height=12),
                    ft.Container(
                        content=selected_files,
                        padding=12,
                        border_radius=10,
                        bgcolor=ft.Colors.WHITE,
                        border=ft.border.all(1, ft.Colors.BLUE_GREY_100),
                    ),
                    ft.Container(height=12),
                    ft.Container(
                        content=extraction_stats,
                        padding=12,
                        border_radius=10,
                        bgcolor=ft.Colors.WHITE,
                        border=ft.border.all(1, ft.Colors.BLUE_GREY_100),
                    ),
                ],
                spacing=4,
            ),
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.BLUE_GREY_50,
        )
    )

    page.update()
