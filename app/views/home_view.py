import flet as ft

from services.extraction_service import run_extraction


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
        if e.files:
            selected_file_paths.clear()
            for file in e.files:
                if file.path:
                    selected_file_paths.append(file.path)

            selected_files.value = "\n".join(file.name for file in e.files)
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

        selected_files.value = f"✅ Extracted {len(results)} rows"
        print(results[:5])  # debug
        page.update()

    picker = ft.FilePicker(on_result=on_pick_result)
    page.overlay.append(picker)

    page.add(
        ft.Text("Imarisha Automark", size=28, weight=ft.FontWeight.BOLD),
        ft.Row(
            controls=[
                ft.ElevatedButton(
                    "Select file(s)",
                    on_click=lambda _: picker.pick_files(allow_multiple=True),
                ),
                mode_dropdown,
                ft.ElevatedButton("Run extraction", on_click=run_clicked),
                ft.ElevatedButton("Open review"),
            ]
        ),
        ft.Container(height=12),
        selected_files,
    )

    page.update()
