
# File picker
picker = ft.FilePicker(on_result=on_pick_result)
page.file_picker = picker
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
        res = run_extraction(
            file_path=path,
            student_id_mode=mode_dropdown.value or "auto",
        )
        results.extend(res)

    selected_files.value = f"✅ Extracted {len(results)} rows"
    print(results[:5])
    page.update()
