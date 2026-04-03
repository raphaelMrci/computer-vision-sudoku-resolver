PYTHON := $(shell if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"; else echo "python3"; fi)
PORT := 8080
IMAGE_NAME := sudoku-vision-api

.PHONY: build api solve config loop

build:
	docker build -t $(IMAGE_NAME):latest .

api:
	$(PYTHON) -m api.app

solve:
	$(PYTHON) -m scripts.live_solve

config:
	$(PYTHON) -m scripts.live_solve --save-config

loop:
	$(PYTHON) -m scripts.live_solve --loop
