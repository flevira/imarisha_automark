
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

    def select_files_clicked(_: ft.ControlEvent) -> None:
        print("[home_view] Select file(s) button clicked")
        try:
            print("[home_view] Calling FilePicker.pick_files()")
            picker.pick_files(
                dialog_title="Choose answer sheet files",
                allow_multiple=True,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            )
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            selected_files.value = f"⚠️ Could not open file browser: {exc}"
            page.update()

    def select_files_fallback_clicked(_: ft.ControlEvent) -> None:
        print("[home_view] Fallback file dialog button clicked")
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
                normalized_paths = [str(Path(path).expanduser().resolve()) for path in selected]
                _set_selected_paths(normalized_paths)
                return

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
            normalized_paths = [str(Path(path).expanduser().resolve()) for path in selected]
            _set_selected_paths(normalized_paths)
        except Exception as exc:  # pragma: no cover - UI fallback
            selected_files.value = f"⚠️ Fallback file dialog failed: {exc}"
            page.update()

    def run_clicked(_: ft.ControlEvent) -> None:
        if not selected_file_paths:
            selected_files.value = "⚠️ No file selected"
            page.update()
            return

        results = run_extraction(
            file_paths=selected_file_paths,
            student_id_mode=mode_dropdown.value or "auto",
        )

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
        print(results[:5])  # debug
        page.update()

    page.add(
        ft.Text("Imarisha Automark", size=28, weight=ft.FontWeight.BOLD),
        ft.Row(
            controls=[
                ft.ElevatedButton("Select file(s)", on_click=select_files_clicked),
                ft.ElevatedButton("Select file(s) (fallback)", on_click=select_files_fallback_clicked),
                mode_dropdown,
                ft.ElevatedButton("Run extraction", on_click=run_clicked),
                ft.ElevatedButton("Open review"),
            ]
        ),
        ft.Container(height=12),
        selected_files,
    )

    page.update()
