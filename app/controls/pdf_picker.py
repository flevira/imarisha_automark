"""Native macOS single-PDF picker for Flet desktop's picker workaround."""
import subprocess


def choose_macos_pdf():
    result = subprocess.run(
        ['osascript', '-e', 'return POSIX path of (choose file with prompt "Choose answer-sheet PDF" of type {"com.adobe.pdf"} without multiple selections allowed)'],
        capture_output=True, text=True, check=False)
    if result.returncode:
        if '(-128)' in result.stderr or 'user canceled' in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr.strip() or 'The macOS file dialog could not open')
    return result.stdout.removesuffix('\n') or None


def choose_macos_manifest():
    result=subprocess.run(['osascript','-e',
        'return POSIX path of (choose file with prompt "Import AutoMark manifest" of type {"public.json"} without multiple selections allowed)'],
        capture_output=True,text=True,check=False)
    if result.returncode:
        if '(-128)' in result.stderr or 'user canceled' in result.stderr.lower():return None
        raise RuntimeError(result.stderr.strip() or 'The macOS file dialog could not open')
    return result.stdout.removesuffix('\n') or None
