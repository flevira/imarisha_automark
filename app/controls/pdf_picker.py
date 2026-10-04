"""Native macOS PDF/image multi-file picker for Flet desktop's picker workaround."""
import subprocess


def choose_macos_pdf():
    result = subprocess.run(
        ['osascript', '-e', 'set chosen to choose file with prompt "Choose answer-sheet PDFs or images" of type {"com.adobe.pdf", "public.image"} with multiple selections allowed',
         '-e', 'set paths to ""', '-e', 'repeat with f in chosen',
         '-e', 'set paths to paths & POSIX path of f & linefeed', '-e', 'end repeat', '-e', 'return paths'],
        capture_output=True, text=True, check=False)
    if result.returncode:
        if '(-128)' in result.stderr or 'user canceled' in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr.strip() or 'The macOS file dialog could not open')
    return [path for path in result.stdout.splitlines() if path] or None


def choose_macos_manifest():
    result=subprocess.run(['osascript','-e',
        'return POSIX path of (choose file with prompt "Import AutoMark manifest" of type {"public.json"} without multiple selections allowed)'],
        capture_output=True,text=True,check=False)
    if result.returncode:
        if '(-128)' in result.stderr or 'user canceled' in result.stderr.lower():return None
        raise RuntimeError(result.stderr.strip() or 'The macOS file dialog could not open')
    return result.stdout.removesuffix('\n') or None
