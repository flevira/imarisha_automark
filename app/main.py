import flet as ft
from database.db import init_db
from app.views.home_view import build_home_view

def main(page: ft.Page) -> None:
    page.title = "Imarisha Automark"
    page.window_width = 1200
    page.window_height = 800
    page.padding = 16

    init_db()
    build_home_view(page)

ft.app(target=main)
