import flet as ft

def build_home_view(page: ft.Page) -> None:
    page.controls.clear()

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
            selected_files.value = "\n".join(f.path or f.name for f in e.files)
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
                ft.ElevatedButton("Run extraction"),
                ft.ElevatedButton("Open review"),
            ]
        ),
        ft.Container(height=12),
        selected_files,
    )

    page.update()
