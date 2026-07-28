#!/usr/bin/env python3
"""
Capture real Gradio UI screenshots with actual YOLO26 detection results.

Uses Playwright to:
1. Open the HapticGuide Gradio UI
2. Upload Pexels stock photos
3. Click Detect to run real YOLO26 inference
4. Screenshot the detection results (both full UI and cropped overlay)

Requires: Gradio server running at http://localhost:7860
"""

import asyncio
import time
from pathlib import Path

from playwright.async_api import async_playwright


GRADIO_URL = "http://localhost:7860"
OUTPUT_DIR = Path("docs/screenshots")
PHOTOS = {
    "scanning": OUTPUT_DIR / "_scene_scanning.jpg",
    "tracking": OUTPUT_DIR / "_scene_tracking.jpg",
    "locked": OUTPUT_DIR / "_scene_locked.jpg",
}

VIEWPORT = {"width": 1440, "height": 900}
SCALE = 2  # Retina


async def detect_and_capture(page, photo_path: Path, mode_name: str):
    """Upload a photo, run detection, and capture screenshots."""
    print(f"\n--- {mode_name.upper()} mode: {photo_path.name} ---")

    # Reload for clean state
    await page.goto(GRADIO_URL, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(5000)

    # Find and use the file upload input
    file_inputs = page.locator('input[type="file"]')
    file_count = await file_inputs.count()
    print(f"  File inputs found: {file_count}")

    if file_count == 0:
        print("  ERROR: No file input found!")
        return

    # Upload the photo
    await file_inputs.first.set_input_files(str(photo_path))
    await page.wait_for_timeout(2000)
    print(f"  Uploaded photo")

    # Set target to "cell phone" (should be default, but ensure it)
    try:
        # Gradio dropdown for target object
        target_dd = page.locator("label:has-text('Target Object')")
        if await target_dd.count() > 0:
            # Try clicking the dropdown and selecting
            input_el = target_dd.locator("input").first
            await input_el.fill("cell phone")
            await page.wait_for_timeout(500)
    except Exception:
        pass  # Default is likely already "cell phone"

    # Click Detect button
    detect_btn = page.get_by_role("button", name="Detect")
    if await detect_btn.count() == 0:
        detect_btn = page.locator("button", has_text="Detect")

    if await detect_btn.count() > 0:
        await detect_btn.first.click()
        print(f"  Clicked Detect")
    else:
        print(f"  ERROR: No Detect button found!")
        return

    # Wait for detection to finish
    print(f"  Running YOLO26 inference (waiting for result)...")
    # Wait for the result image to load - check for img in output area
    for i in range(24):  # up to 120 seconds
        await page.wait_for_timeout(5000)
        # Check if there are images in the result panel
        result_imgs = page.locator("img[src^='data:image'], img[src^='/file']")
        count = await result_imgs.count()
        if count >= 2:
            print(f"  Detection result appeared ({count} images on page)")
            break
        print(f"  ... waiting ({(i+1)*5}s)")

    await page.wait_for_timeout(2000)  # Extra rendering time

    # Screenshot the full Gradio page
    out_path = OUTPUT_DIR / f"gradio_{mode_name}.png"
    await page.screenshot(path=str(out_path), full_page=False)
    print(f"  Saved: {out_path.name}")

    # Try to extract just the detection overlay image
    # Gradio puts results in img tags inside specific containers
    try:
        all_imgs = page.locator("img")
        img_count = await all_imgs.count()
        # The detection result is typically the second large image
        # (first is the upload preview, second is the detection output)
        for idx in range(img_count - 1, -1, -1):
            img = all_imgs.nth(idx)
            box = await img.bounding_box()
            if box and box["width"] > 200:
                display_out = OUTPUT_DIR / f"display_{mode_name}.png"
                await img.screenshot(path=str(display_out))
                print(f"  Saved detection overlay: {display_out.name} ({box['width']:.0f}x{box['height']:.0f})")
                break
    except Exception as e:
        print(f"  Could not crop result: {e}")


async def capture_screenshots():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=SCALE,
        )
        page = await context.new_page()

        print("Loading HapticGuide Gradio UI...")
        await page.goto(GRADIO_URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        title = await page.title()
        print(f"  Page title: {title}")

        # Verify it's the right app
        if "HapticGuide" not in title and "haptic" not in (await page.content()).lower()[:2000]:
            print("  WARNING: Page doesn't appear to be HapticGuide Gradio!")
            # Try anyway

        # Process each photo
        for mode_name, photo_path in PHOTOS.items():
            if not photo_path.exists():
                print(f"  SKIP {mode_name}: photo not found")
                continue
            await detect_and_capture(page, photo_path, mode_name)

        # Final best screenshot for README (locked mode)
        print(f"\n--- Final README Gradio screenshot ---")
        # Reload and redo locked detection
        await page.goto(GRADIO_URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        locked_photo = PHOTOS.get("locked")
        if locked_photo and locked_photo.exists():
            file_inputs = page.locator('input[type="file"]')
            await file_inputs.first.set_input_files(str(locked_photo))
            await page.wait_for_timeout(2000)

            detect_btn = page.get_by_role("button", name="Detect")
            if await detect_btn.count() == 0:
                detect_btn = page.locator("button", has_text="Detect")
            await detect_btn.first.click()

            # Wait for result
            for i in range(24):
                await page.wait_for_timeout(5000)
                result_imgs = page.locator("img[src^='data:image'], img[src^='/file']")
                if await result_imgs.count() >= 2:
                    break
            await page.wait_for_timeout(2000)

        await page.screenshot(path=str(OUTPUT_DIR / "gradio_ui.png"), full_page=False)
        print(f"  Saved: gradio_ui.png")

        await browser.close()
        print(f"\nDone! Screenshots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(capture_screenshots())
