# Computer Vision Sudoku Solver

Vision-only Sudoku solver for `https://sudoku.com/` with a **server/client** architecture:

- **Solver API (server)**: OpenCV + Tesseract + Sudoku solver, exposed over HTTP.
- **Live client**: captures screenshots, sends to API, autofills the browser.

No DOM parsing is used for grid/digit perception.

## Architecture

- `api/app.py`: HTTP API (`/predict`, `/predict_debug`)
- `src/pipeline.py`: detect -> OCR -> solve
- `scripts/live_solve.py`: local client (capture + API call + typing)
- `Dockerfile`: dockerized API server

## Quick start (local dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run API locally:

```bash
make api
```

Run client locally:

```bash
make live-solve
```

## Dockerized solver API

Build and run:

```bash
make docker-build
make docker-run
```

This starts the solver API at `http://localhost:8080`.

Health check:

```bash
curl http://localhost:8080/health
```

Predict from an image:

```bash
curl -X POST "http://localhost:8080/predict" -F "file=@/path/to/image.png"
```

## Client usage (screenshots + auto resolution)

The client runs on host machine (needs screen and keyboard/mouse access), calls API:

```bash
.venv/bin/python -m scripts.live_solve
```

Behavior is fixed by design:
- API endpoint: `http://localhost:8080/predict`
- API timeout: `120s`
- minimum clues: `17` (minimum valid Sudoku clues)
- interactive calibration on each run

Loop mode (infinite benchmark):

```bash
.venv/bin/python -m scripts.live_solve --loop
```

Save config once (grid + loop buttons), then reuse automatically:

```bash
.venv/bin/python -m scripts.live_solve --save-config
```

In loop mode, the client asks for:
- screenshot area (grid),
- `New game` and `Extreme` button positions for SUCCESS screen,
- `New game` and `Extreme` button positions for FAILED screen.

Then `--loop` runs forever, tracking:
- launched games,
- successes,
- failures,
- success rate (%).

Safety:
- move mouse to top-left screen corner (pyautogui fail-safe),
- or press `Ctrl+C`,
- auto-stop after too many consecutive failures.

## Other useful commands

- Save debug overlay from API:
  - `make debug-image IMAGE="/path/to/image.png" OUT="debug.png"`
- Generate OCR-prefilled manifest:
  - `make generate-manifest MANIFEST_INPUT="data/raw_screenshots" MANIFEST_OUT="data/manifest_autogen.json"`
- Capture screenshot dataset loop:
  - `make capture-dataset CAPTURE_OUT="data/raw_screenshots"`

## Notes

- Docker image already installs `tesseract-ocr` for server-side OCR.
- On macOS, grant **Screen Recording** and **Accessibility** to terminal/IDE for live client automation.
