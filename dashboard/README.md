# M6 — Investigation Dashboard

Drop this whole `dashboard/` folder into the root of `OilSpill_SIH`, so it sits
next to `outputs/`, `modules/`, etc.

## 1. Backend

```bash
cd dashboard/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

By default it assumes it's sitting at `OilSpill_SIH/dashboard/backend/main.py`
and walks up to find `outputs/`. If you put it somewhere else, set:

```bash
OILSPILL_ROOT=/path/to/OilSpill_SIH uvicorn main:app --reload
```

Check it's finding your data:
```
http://127.0.0.1:8000/api/health
```

## 2. Frontend

No build step — it's a single HTML file using Leaflet from a CDN.
Just open `dashboard/frontend/index.html` directly in your browser
(double-click it, or right-click → Open With → Browser), with the
backend running in the background.

If you want it served properly instead of via `file://`, from
`dashboard/frontend/`:
```bash
python3 -m http.server 5500
```
then visit `http://127.0.0.1:5500`.

## What it shows

- Dropdown at the top switches between every `slick_id` that has ranked
  vessels (the drift → AIS → attribution chain).
- Map: spill polygon (if available), estimated origin, hindcast/forecast
  drift path, and every candidate vessel's track (dimmed unless selected).
- Right panel: spill summary + ranked suspect list by default; clicking a
  vessel shows its score breakdown, plain-English explanation, and any
  AIS gaps.

## Known limitation (not a dashboard bug)

Not every `slick_id` will have a matching spill polygon — `spill.geometry`
comes from M2's `spill_metadata.json`, which currently doesn't cover the
same dataset as the drift/AIS/scoring chain. The dashboard already handles
this gracefully (it just shows the origin marker and drift path without a
polygon in that case), so you don't need to do anything about it right now.