"""Build project/Report/Technical_Report.pdf from Technical_Report.md:
pandoc (Markdown -> self-contained HTML with the figures embedded) + headless
Chrome/Edge (HTML -> PDF). Needs pandoc and a Chromium browser on PATH or in the
default install locations.

    python scripts/build_report.py
"""
import os
import shutil
import subprocess
import sys

REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project", "Report")
BROWSERS = [
    "chrome", "google-chrome", "chromium", "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def browser():
    for b in BROWSERS:
        path = shutil.which(b) or (b if os.path.exists(b) else None)
        if path:
            return path
    sys.exit("no Chrome/Edge found")


def main():
    html = os.path.join(REPORT, "Technical_Report.html")
    pdf = os.path.join(REPORT, "Technical_Report.pdf")
    subprocess.run(["pandoc", "Technical_Report.md", "-o", html, "--standalone", "--embed-resources",
                    "--css", "report.css", "--resource-path", REPORT, "--metadata", "lang=en"],
                   cwd=REPORT, check=True)
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + pdf, "file:///" + html.replace("\\", "/")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(html)
    print("wrote", pdf, "(%.1f MB)" % (os.path.getsize(pdf) / 1e6))


if __name__ == "__main__":
    main()
