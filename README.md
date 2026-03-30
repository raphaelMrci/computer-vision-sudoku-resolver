# Computer Vision Sudoku Solver

Vision-only Sudoku solver for `https://sudoku.com/` with a simplified stack:

- grid detection with OpenCV,
- OCR with **Tesseract only**,
- Sudoku solve with backtracking,
- browser autofill via `pyautogui`.

No DOM parsing is used for grid perception.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main commands

- Run API: `make api`
- Run live solver: `make live-solve`
- Save debug image from API: `make debug-image IMAGE="/path/to/image.png" OUT="debug.png"`
- Generate OCR-prefilled manifest from screenshots: `make generate-manifest MANIFEST_INPUT="data/raw_screenshots" MANIFEST_OUT="data/manifest_autogen.json"`
- Capture screenshots loop: `make capture-dataset CAPTURE_OUT="data/raw_screenshots"`

## Live solve

```bash
.venv/bin/python -m scripts.live_solve --min-clues 17
```

Optional direct calibration values:

```bash
.venv/bin/python -m scripts.live_solve --x 320 --y 195 --size 517 --min-clues 17
```

## API

Start:

```bash
make api
```

Predict:

```bash
curl -X POST "http://localhost:8080/predict" -F "file=@/path/to/image.png"
```

Predict with debug overlay:

```bash
curl -X POST "http://localhost:8080/predict_debug" -F "file=@/path/to/image.png"
```

## Notes

- Tesseract binary must be installed and available in `PATH` (or in standard macOS Homebrew paths).
- On macOS, grant **Screen Recording** and **Accessibility** permissions for terminal/IDE when using live mode.
