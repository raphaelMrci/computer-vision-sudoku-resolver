PYTHON := $(shell if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"; else echo "python3"; fi)
PORT := 8080
IMAGE_NAME := sudoku-vision-api
IMAGE ?=
OUT ?= debug_from_api.png
X ?=
Y ?=
SIZE ?=
CAPTURE_OUT ?= data/raw_screenshots
CAPTURE_INTERVAL ?= 3
CAPTURE_PREFIX ?= sudoku
MANIFEST_INPUT ?= data/raw_screenshots
MANIFEST_OUT ?= data/manifest_autogen.json

.PHONY: install api demo debug-image generate-manifest live-solve capture-dataset docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements.txt

api:
	$(PYTHON) -m api.app

demo:
	$(PYTHON) src/pipeline.py

debug-image:
	$(PYTHON) -m scripts.save_debug_image --image "$(IMAGE)" --out "$(OUT)"

generate-manifest:
	$(PYTHON) -m scripts.generate_manifest --input-dir "$(MANIFEST_INPUT)" --out "$(MANIFEST_OUT)" --prefill-ocr

live-solve:
	$(PYTHON) -m scripts.live_solve $(if $(X),--x "$(X)") $(if $(Y),--y "$(Y)") $(if $(SIZE),--size "$(SIZE)")

capture-dataset:
	$(PYTHON) -m scripts.capture_dataset_loop --out-dir "$(CAPTURE_OUT)" --interval $(CAPTURE_INTERVAL) --prefix "$(CAPTURE_PREFIX)" --countdown

docker-build:
	docker build -t $(IMAGE_NAME):latest .

docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE_NAME):latest
