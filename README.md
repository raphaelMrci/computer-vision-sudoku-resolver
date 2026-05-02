# Computer Vision Sudoku Solver

---

## 1. Set up the environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
make api
```

---

## 2. Create a new configuration

```bash
make config
```

Configuration is saved to `data/live_solve_config.json`.

---

## 3. Solve a Sudoku

Terminal 1 (API):

```bash
make api
```

Terminal 2 (client):

```bash
make solve
```

---

## 4. Solve in a loop

After running **`make config`** at least once:

```bash
make loop
```
