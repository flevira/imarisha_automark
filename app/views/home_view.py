
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

    def select_files_clicked(_: ft.ControlEvent) -> None:
        print("[home_view] Select file(s) button clicked")
        try:
            print("[home_view] Calling FilePicker.pick_files()")
            picker.pick_files(
                dialog_title="Choose answer sheet files",
                allow_multiple=True,
                file_type=ft.FilePickerFileType.ANY,
            )
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            selected_files.value = f"⚠️ Could not open file browser: {exc}"
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

    page.add(
        ft.Text("Imarisha Automark", size=28, weight=ft.FontWeight.BOLD),
        ft.Row(
            controls=[
                ft.ElevatedButton("Select file(s)", on_click=select_files_clicked),
                mode_dropdown,
                ft.ElevatedButton("Run extraction", on_click=run_clicked),
                ft.ElevatedButton("Open review"),
            ]
        ),
        ft.Container(height=12),
        selected_files,
    )

    page.update()
