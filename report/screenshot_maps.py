"""Save PNG snapshots of the folium HTML maps for the PDF report.

Usage: python report/screenshot_maps.py results/maps/pred_00000040.html out.png [--vendor DIR]

--vendor: folder with local copies of leaflet / jquery / bootstrap /
awesome-markers (from npm). Only needed on a machine without internet; the
CDN requests are then answered from that folder. Background tiles still need
internet, without them the map is drawn on a grey background.
"""
import argparse
import asyncio
import os

from playwright.async_api import async_playwright

LOCAL = {
    "leaflet.js": "leaflet/dist/leaflet.js",
    "leaflet.css": "leaflet/dist/leaflet.css",
    "jquery-3.7.1.min.js": "jquery/dist/jquery.min.js",
    "bootstrap.bundle.min.js": "bootstrap/dist/js/bootstrap.bundle.min.js",
    "bootstrap.min.css": "bootstrap/dist/css/bootstrap.min.css",
    "leaflet.awesome-markers.js": "awesome/dist/leaflet.awesome-markers.js",
    "leaflet.awesome-markers.css": "awesome/dist/leaflet.awesome-markers.css",
}


async def shoot(html, out, vendor, w, h, zoom=None):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": w, "height": h})

        async def handle(route):
            url = route.request.url
            name = url.split("?")[0].rsplit("/", 1)[-1]
            if vendor and name in LOCAL and os.path.exists(os.path.join(vendor, LOCAL[name])):
                ctype = "text/css" if name.endswith(".css") else "application/javascript"
                await route.fulfill(path=os.path.join(vendor, LOCAL[name]), content_type=ctype)
            elif url.startswith("file:"):
                await route.continue_()
            elif vendor:
                await route.abort()
            else:
                await route.continue_()

        await pg.route("**/*", handle)
        await pg.goto("file://" + os.path.abspath(html))
        await pg.wait_for_timeout(2500)
        if zoom:
            lat, lon, z = zoom.split(",")
            await pg.evaluate(f"window[document.querySelector('.folium-map').id].setView([{lat},{lon}],{z})")
            await pg.wait_for_timeout(1500)
        await pg.screenshot(path=out)
        await b.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("out")
    ap.add_argument("--vendor", default=None)
    ap.add_argument("--w", type=int, default=1200)
    ap.add_argument("--h", type=int, default=760)
    ap.add_argument("--zoom", default=None, help="lat,lon,zoom to set the view before the screenshot")
    a = ap.parse_args()
    asyncio.run(shoot(a.html, a.out, a.vendor, a.w, a.h, a.zoom))
