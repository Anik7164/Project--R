"""
Draws the app icon (the pitch mark of the page's header) in the sizes the
web app, the desktop app and the iPhone app need. Run once; the results
are in the repository.

    python webapp/make_icons.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BACKGROUND, PITCH = (11, 18, 16), (53, 194, 106)


def icon(size, margin=0.2):
    """A green pitch circle with its halfway line and centre spot."""
    big = size * 4                      # drawn large, shrunk: smooth edges
    image = Image.new("RGB", (big, big), BACKGROUND)
    draw = ImageDraw.Draw(image)
    edge = big * margin
    draw.ellipse((edge, edge, big - edge, big - edge), fill=PITCH)
    line = big * 0.035
    draw.rectangle((big / 2 - line / 2, edge, big / 2 + line / 2, big - edge), fill=BACKGROUND)
    spot = big * 0.085
    draw.ellipse((big / 2 - spot, big / 2 - spot, big / 2 + spot, big / 2 + spot), fill=BACKGROUND)
    return image.resize((size, size), Image.LANCZOS)


def main():
    static = ROOT / "webapp" / "static"
    for size in (180, 192, 512):
        icon(size).save(static / f"icon-{size}.png", optimize=True)
    icon(256).save(ROOT / "desktop" / "icon.ico", sizes=[(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)])
    icon(512).save(ROOT / "desktop" / "icon.png", optimize=True)
    icon(1024).save(ROOT / "mobile" / "assets" / "icon.png", optimize=True)   # iOS: no transparency
    # The app's launch screen: the mark, small, on the page's background.
    splash = Image.new("RGB", (2732, 2732), BACKGROUND)
    mark = icon(520, margin=0.0)
    mask = Image.new("L", (520 * 4, 520 * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 520 * 4, 520 * 4), fill=255)
    splash.paste(mark, (1106, 1106), mask.resize((520, 520), Image.LANCZOS))
    splash.save(ROOT / "mobile" / "assets" / "splash.png", optimize=True)
    print("icons written")


if __name__ == "__main__":
    main()
