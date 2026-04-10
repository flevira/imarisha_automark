from __future__ import annotations

import flet as ft

from services.extraction_service import run_extraction


def build_home_view(page: ft.Page) -> None:
    selected_file_paths: list[str] = []

    selected_files = ft.Text("No file selected")
    mode_dropdown = ft.Dropdown(
        label="Student ID mode",
        value="auto",
        options=[
            ft.dropdown.Option("auto"),
            ft.dropdown.Option("qr"),
            ft.dropdown.Option("text"),
        ],
        width=200,
    )

    def on_pick_result(event: ft.FilePickerResultEvent) -> None:
        nonlocal selected_file_paths
        files = event.files or []
        selected_file_paths = [f.path for f in files if f.path]

        if selected_file_paths:
            selected_files.value = f"Selected {len(selected_file_paths)} file(s)"
        else:
            selected_files.value = "No file selected"

        page.update()

    picker = ft.FilePicker(on_result=on_pick_result)
    page.overlay.append(picker)

    def select_files_clicked(_: ft.ControlEvent) -> None:
        picker.pick_files(
            dialog_title="Choose answer sheet files",
            allow_multiple=True,
            file_type=ft.FilePickerFileType.ANY,
        )

    def run_clicked(_: ft.ControlEvent) -> None:
        if not selected_file_paths:
            selected_files.value = "⚠️ No file selected"
            page.update()
            return

        results = []
        for path in selected_file_paths:
            results.extend(
                run_extraction(
                    file_path=path,
                    student_id_mode=mode_dropdown.value or "auto",
                )
            )

        selected_files.value = f"✅ Extracted {len(results)} rows"
        page.update()

    page.controls.clear()
    page.add(
        ft.Column(
            controls=[
                ft.Text("Imarisha Automark", size=24, weight=ft.FontWeight.BOLD),
                mode_dropdown,
                ft.Row(
                    controls=[
                        ft.ElevatedButton("Select files", on_click=select_files_clicked),
                        ft.ElevatedButton("Run extraction", on_click=run_clicked),
                    ],
                ),
                selected_files,
            ],
            spacing=12,
        )
    )
    page.update()
