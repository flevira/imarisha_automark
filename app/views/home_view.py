
import flet as ft
import json
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

from services.extraction_service import run_extraction

ENABLE_TEMP_CSV_EXPORT = True


def build_home_view(page: ft.Page) -> None:
    page.controls.clear()

    selected_file_paths: list[str] = []
    selected_files = ft.Text("No file selected")
    extraction_stats = ft.Text("", selectable=True, visible=False)
    extraction_progress = ft.Row(
        controls=[
            ft.ProgressRing(width=20, height=20, stroke_width=3),
            ft.Text(
                "Extraction in progress… Please wait while we process your files.",
                color=ft.Colors.BLUE_GREY_700,
                weight=ft.FontWeight.W_500,
            ),
        ],
        visible=False,
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
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
    run_button = ft.ElevatedButton(
        "Run extraction",
        icon=ft.Icons.PLAY_ARROW,
        on_click=lambda _: None,
        width=220,
        height=44,
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
        if not selected_file_paths:
            selected_files.value = "⚠️ No file selected"
            page.update()
            return

        run_button.disabled = True
        extraction_progress.visible = True
        selected_files.value = "⏳ Extraction in progress..."
        page.update()

        try:
            results = run_extraction(
                file_paths=selected_file_paths,
                student_id_mode=mode_dropdown.value or "auto",
            )
        finally:
            run_button.disabled = False
            extraction_progress.visible = False

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
        brightness_values = [
            float(page.get("scan_brightness_mean"))
            for page in page_quality.values()
            if isinstance(page.get("scan_brightness_mean"), (int, float))
        ]
        contrast_values = [
            float(page.get("scan_contrast_std"))
            for page in page_quality.values()
            if isinstance(page.get("scan_contrast_std"), (int, float))
        ]
        blur_values = [
            float(page.get("scan_blur_laplacian_var"))
            for page in page_quality.values()
            if isinstance(page.get("scan_blur_laplacian_var"), (int, float))
        ]
        skew_values = [
            float(page.get("scan_estimated_skew_deg"))
            for page in page_quality.values()
            if isinstance(page.get("scan_estimated_skew_deg"), (int, float))
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
        avg_brightness = (
            round(sum(brightness_values) / len(brightness_values), 1) if brightness_values else None
        )
        avg_contrast = round(sum(contrast_values) / len(contrast_values), 1) if contrast_values else None
        avg_blur = round(sum(blur_values) / len(blur_values), 1) if blur_values else None
        avg_skew = round(sum(skew_values) / len(skew_values), 2) if skew_values else None

        section_lines = [f"- {name}: {count}" for name, count in sorted(sections.items())]
        quality_lines = (
            [
                f"- Pages analyzed for quality: {len(page_quality)}",
                f"- Average scan quality score: {avg_quality}/100",
                f"- Combined brightness mean: {avg_brightness if avg_brightness is not None else 'n/a'}",
                f"- Combined contrast std: {avg_contrast if avg_contrast is not None else 'n/a'}",
                f"- Combined blur proxy: {avg_blur if avg_blur is not None else 'n/a'}",
                f"- Combined estimated skew: {avg_skew if avg_skew is not None else 'n/a'}°",
                f"- Good scans: {good_pages}",
                f"- Fair scans: {fair_pages}",
                f"- Poor scans: {poor_pages}",
            ]
            if page_quality
            else ["- Scan quality unavailable"]
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
                "- Section counts:",
                *(section_lines if section_lines else ["- (none)"]),
            ]
        )
        extraction_stats.visible = True
        print(results[:5])  # debug
        page.update()

    run_button.on_click = run_clicked

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
                    ft.Text(
                        "Accepted files: PDF, PNG, JPG/JPEG, TIF/TIFF, BMP, WEBP. "
                        "Recommended quality: 300+ DPI, flat/no tilt, well lit, and in sharp focus.",
                        size=12,
                        color=ft.Colors.BLUE_GREY_600,
                    ),
                    ft.Text(
                        "Powered by Science Innovation Limited",
                        size=11,
                        italic=True,
                        color=ft.Colors.BLUE_GREY_500,
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
                                content=run_button,
                            ),
                            ft.Container(
                                col={"sm": 12, "md": 2},
                                content=ft.OutlinedButton(
                                    "Open review",
                                    icon=ft.Icons.RATE_REVIEW,
                                    disabled=True,
                                    tooltip="Coming soon: review workflow",
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
                        content=extraction_progress,
                        padding=8,
                        border_radius=10,
                        bgcolor=ft.Colors.BLUE_50,
                    ),
                    ft.Container(height=8),
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
