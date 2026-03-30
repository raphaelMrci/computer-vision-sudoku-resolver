PYTHON := $(shell if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"; else echo "python3"; fi)
PORT := 8080
IMAGE_NAME := sudoku-vision-api
IMAGE ?=
OUT ?= debug_from_api.png
OCR_MODE ?= advanced
X ?=
Y ?=
SIZE ?=
MANIFEST ?= docs/benchmark-manifest.example.json
BENCH_OUT ?= artifacts/ocr_benchmark.json
YOLO_DATA ?=
YOLO_MODEL ?= yolov8n.pt
YOLO_EPOCHS ?= 50
YOLO_IMGSZ ?= 640
YOLO_BATCH ?= 16
YOLO_DEVICE ?= cpu
OCR_DATA_DIR ?= data/ocr_cnn
OCR_CNN_OUT ?= models/ocr_cnn.pt
OCR_CNN_EPOCHS ?= 12
OCR_CNN_BATCH ?= 64
OCR_CNN_LR ?= 0.001
CAPTURE_OUT ?= data/raw_screenshots
CAPTURE_INTERVAL ?= 3
CAPTURE_PREFIX ?= sudoku
MANIFEST_INPUT ?= data/raw_screenshots
MANIFEST_OUT ?= data/manifest_autogen.json
FILTERED_MANIFEST_OUT ?= data/manifest_filtered.json
TARGET_REPORT_OUT ?= artifacts/benchmark_targets.md

.PHONY: install api train infer demo debug-image benchmark-ocr benchmark-targets generate-manifest filter-manifest build-ocr-dataset train-ocr-cnn train-yolo live-solve capture-dataset docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements.txt

api:
	$(PYTHON) -m api.app

train:
	$(MAKE) train-yolo

infer:
	@echo "TODO: add local inference command. IMAGE=$(IMAGE)"

demo:
	$(PYTHON) src/pipeline.py

debug-image:
	$(PYTHON) -m scripts.save_debug_image --image "$(IMAGE)" --out "$(OUT)" --ocr-mode "$(OCR_MODE)"

benchmark-ocr:
	$(PYTHON) -m scripts.benchmark_ocr --manifest "$(MANIFEST)" --out "$(BENCH_OUT)"

benchmark-targets:
	$(PYTHON) -m scripts.benchmark_targets --benchmark "$(BENCH_OUT)" --out "$(TARGET_REPORT_OUT)"

generate-manifest:
	$(PYTHON) -m scripts.generate_manifest --input-dir "$(MANIFEST_INPUT)" --out "$(MANIFEST_OUT)" --prefill-ocr --ocr-mode "$(OCR_MODE)"

filter-manifest:
	$(PYTHON) -m scripts.filter_manifest --manifest "$(MANIFEST_OUT)" --out "$(FILTERED_MANIFEST_OUT)" --require-grid

build-ocr-dataset:
	$(PYTHON) -m scripts.build_ocr_dataset --manifest "$(MANIFEST)" --out-dir "$(OCR_DATA_DIR)"

train-ocr-cnn:
	$(PYTHON) -m scripts.train_ocr_cnn --data-dir "$(OCR_DATA_DIR)" --epochs $(OCR_CNN_EPOCHS) --batch-size $(OCR_CNN_BATCH) --lr $(OCR_CNN_LR) --out "$(OCR_CNN_OUT)"

train-yolo:
	$(PYTHON) -m scripts.train_yolo --data "$(YOLO_DATA)" --model "$(YOLO_MODEL)" --epochs $(YOLO_EPOCHS) --imgsz $(YOLO_IMGSZ) --batch $(YOLO_BATCH) --device "$(YOLO_DEVICE)"

live-solve:
	$(PYTHON) -m scripts.live_solve $(if $(X),--x "$(X)") $(if $(Y),--y "$(Y)") $(if $(SIZE),--size "$(SIZE)")

capture-dataset:
	$(PYTHON) -m scripts.capture_dataset_loop --out-dir "$(CAPTURE_OUT)" --interval $(CAPTURE_INTERVAL) --prefix "$(CAPTURE_PREFIX)" --countdown

docker-build:
	docker build -t $(IMAGE_NAME):latest .

docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE_NAME):latest
