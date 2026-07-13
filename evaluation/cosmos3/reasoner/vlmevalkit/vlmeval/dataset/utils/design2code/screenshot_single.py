import os

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    error_msg = """
playwright not installed. Please install it with the following commands:
pip install playwright
playwright install-deps
playwright install --no-shell chromium-headless-shell
"""
    print(error_msg)
    exit(1)
import argparse

from PIL import Image


def take_screenshot(url, output_file="screenshot.png", do_it_again=False):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    if os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-gpu",
                    "--no-first-run",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                ],
            )
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)

            # Take the screenshot
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

            browser.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image
        img = Image.new('RGB', (1280, 960), color='white')
        img.save(output_file)


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description='Process two path strings.')

    # Define the arguments
    parser.add_argument('--html', type=str)
    parser.add_argument('--png', type=str)

    # Parse the arguments
    args = parser.parse_args()

    take_screenshot(args.html, args.png, do_it_again=True)
