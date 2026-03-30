from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
import urllib.error
import urllib.request
import uuid


def _encode_multipart_formdata(field_name: str, filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"----sudoku-debug-{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    return body, boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call /predict_debug and save debug PNG.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out", default="debug_from_api.png", help="Output debug PNG path.")
    parser.add_argument("--url", default="http://localhost:8080/predict_debug", help="Predict debug endpoint URL.")
    parser.add_argument("--ocr-mode", default="tesseract", choices=["tesseract"], help="OCR mode.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    out_path = Path(args.out)

    if not image_path.exists():
        print(f"Input image not found: {image_path}")
        return 1

    image_bytes = image_path.read_bytes()
    content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    body, boundary = _encode_multipart_formdata("file", image_path.name, image_bytes, content_type)
    separator = "&" if "?" in args.url else "?"
    endpoint = f"{args.url}{separator}ocr_mode={args.ocr_mode}"
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_bytes = response.read()
            status_code = response.getcode()
    except urllib.error.HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace")
        print(f"Request failed: HTTP {exc.code}")
        print(error_payload)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}")
        return 1

    if status_code != 200:
        print(f"Request failed: HTTP {status_code}")
        print(response_bytes.decode("utf-8", errors="replace"))
        return 1

    payload = json.loads(response_bytes.decode("utf-8"))
    b64 = payload.get("debug_image_base64")
    if not b64:
        print("Response does not contain debug_image_base64.")
        print(json.dumps(payload, indent=2))
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64))

    result = payload.get("result", {})
    status = result.get("status", "unknown")
    clues = result.get("num_clues_detected", "n/a")
    conf = result.get("ocr_confidence_mean", "n/a")
    print(f"Saved debug image to: {out_path}")
    print(f"status={status} clues={clues} ocr_confidence_mean={conf} ocr_mode={args.ocr_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
