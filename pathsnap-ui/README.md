# PathSnap evaluator UI (Laptop 2)

Plain HTML + JavaScript. Leaflet is bundled in `vendor/`, so no CDN is needed;
only the OpenStreetMap background tiles need internet.

```bash
cd pathsnap-ui
python -m http.server 5500
```

Open `http://localhost:5500`, type the API address of Laptop 1
(`http://<laptop-1-ip>:8000`) in the box at the top and press **Connect**
(or set it in `config.js` before the demo).

Flow during the evaluation:

1. `GET /trips` fills the list with the ten evaluation trips.
2. Clicking a trip sends **only its id** to `POST /predict`.
3. The API runs the whole pipeline and scores it on the server.
4. The page draws the returned layers (candidate road network, GPS readings,
   classifier-only route, HMM route) and shows the metrics and timings.

The request log at the bottom left records every call the page makes.
