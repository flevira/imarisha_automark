import flet as ft
from app.views.structured_view import build_structured_view

def main(page: ft.Page) -> None:
    page.title = "Imarisha Automark"
    page.window_width = 1200
    page.window_height = 800
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.Colors.GREY_50

    build_structured_view(page)

ft.app(target=main)
