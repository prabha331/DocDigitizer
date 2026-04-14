import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)

# Point pytesseract at the Windows install location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Language code mapping
# ---------------------------------------------------------------------------
LANGUAGE_MAP = {
    "en": ["en"],
    "hi": ["hi"],
    "ta": ["ta"],
}

# ---------------------------------------------------------------------------
# 1. preprocess_image
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image, handle dark/inverted documents, enhance contrast,
    binarise, and deskew.  Returns cleaned numpy array or original on failure.
    """
    original = None
    try:
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"cv2.imread could not open: {image_path}")

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Detect dark/inverted image (white text on dark background)
        mean_brightness = np.mean(gray)
        if mean_brightness < 127:
            logger.info("Dark image detected (mean=%.1f) — inverting.", mean_brightness)
            gray = cv2.bitwise_not(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Increase contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Deskew via HoughLinesP
        binary = _deskew(binary)

        logger.info("Image preprocessed successfully: %s", image_path)
        return binary

    except Exception as exc:
        logger.error("preprocess_image failed for %s: %s", image_path, exc)
        return original if original is not None else cv2.imread(image_path)


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect dominant line angle with HoughLinesP and rotate the image to
    correct for skew.  Returns the image unchanged if skew < 0.5 degrees.
    """
    # Invert so lines are white on black (HoughLinesP works on bright edges)
    inverted = cv2.bitwise_not(image)
    edges = cv2.Canny(inverted, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:                          # avoid division by zero
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))

    if abs(median_angle) <= 0.5:
        return image                               # skew negligible — skip rotation

    logger.debug("Deskewing image by %.2f degrees", median_angle)
    h, w = image.shape[:2]
    centre = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(centre, median_angle, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ---------------------------------------------------------------------------
# 2. extract_text
# ---------------------------------------------------------------------------

def extract_text(file_path: str, language: str = "en") -> dict:
    """
    Run OCR on an image or PDF.  Tries PaddleOCR first; falls back to
    pytesseract if PaddleOCR raises a runtime error.

    Returns:
      {
        "raw_text":   str,
        "blocks":     [{"text": str, "confidence": float, "bbox": [x1,y1,x2,y2]}],
        "page_count": int,
        "language":   str,
        "engine":     "paddle" | "tesseract",
      }
    """
    lang_codes = LANGUAGE_MAP.get(language, ["en"])
    lang_code = lang_codes[0]

    ext = Path(file_path).suffix.lower()
    all_blocks: list[dict] = []
    page_count = 0
    engine_used = "paddle"

    # Collect images to process (list of numpy BGR arrays)
    images: list[np.ndarray] = []
    if ext == ".pdf":
        logger.info("Processing PDF: %s", file_path)
        pil_pages = convert_from_path(file_path)
        page_count = len(pil_pages)
        for pil_img in pil_pages:
            images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
    else:
        logger.info("Processing image: %s", file_path)
        page_count = 1
        images.append(preprocess_image(file_path))

    tess_lang = {"en": "eng", "hi": "hin", "ta": "tam"}.get(language, "eng")

    # --- Try PaddleOCR ---
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang_code)
        for img in images:
            all_blocks.extend(_run_paddle(ocr, img))
        engine_used = "paddle"
        logger.info("PaddleOCR succeeded with %d blocks.", len(all_blocks))
    except Exception as paddle_err:
        logger.warning("PaddleOCR failed (%s) — falling back to pytesseract.", paddle_err)
        all_blocks = []

    # --- Fallback to tesseract if PaddleOCR returned nothing ---
    if not all_blocks:
        logger.warning("PaddleOCR returned 0 blocks — retrying with tesseract fallback.")
        engine_used = "tesseract"
        preprocessed = preprocess_image(file_path) if ext != ".pdf" else images[0]
        all_blocks = _run_tesseract(preprocessed, tess_lang)
        # For multi-page PDFs, process remaining pages too
        for img in images[1:]:
            all_blocks.extend(_run_tesseract(img, tess_lang))

    raw_text = "\n".join(b["text"] for b in all_blocks)
    logger.info(
        "extract_text complete — engine=%s pages=%d blocks=%d language=%s",
        engine_used, page_count, len(all_blocks), language,
    )

    return {
        "raw_text": raw_text,
        "blocks": all_blocks,
        "page_count": page_count,
        "language": language,
        "engine": engine_used,
    }


def _run_paddle(ocr: PaddleOCR, image: np.ndarray) -> list[dict]:
    """
    Run PaddleOCR on a single numpy image and return a normalised list of
    block dicts:  {"text": str, "confidence": float, "bbox": [x1,y1,x2,y2]}

    Handles both the classic PaddleOCR format and the PP-OCRv5 format:
      Classic:  [ [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],  (text, confidence) ]
      PP-OCRv5: may return objects or nested lists differently
    Each line is parsed independently so one malformed entry never
    crashes the whole page.
    """
    results = ocr.ocr(image)
    blocks: list[dict] = []

    if not results:
        return blocks

    for page_result in results:
        if not page_result:
            continue

        # PP-OCRv5 may wrap the page result in an extra object layer
        if hasattr(page_result, "__dict__"):
            # Object-style result (PP-OCRv5 OCRResult)
            try:
                texts   = getattr(page_result, "rec_texts",  None) or []
                scores  = getattr(page_result, "rec_scores", None) or []
                boxes   = getattr(page_result, "dt_boxes",   None) or []
                for text, conf, box in zip(texts, scores, boxes):
                    text = str(text).strip()
                    if not text:
                        continue
                    try:
                        xs = [float(pt[0]) for pt in box]
                        ys = [float(pt[1]) for pt in box]
                        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    except Exception:
                        bbox = [0, 0, 0, 0]
                    blocks.append({
                        "text": text,
                        "confidence": round(float(conf), 4),
                        "bbox": bbox,
                    })
            except Exception as e:
                logger.debug("_run_paddle object-style parse failed: %s", e)
            continue

        # List/tuple-style result (classic format or PP-OCRv5 list form)
        for line in page_result:
            try:
                if line is None or not hasattr(line, "__len__") or len(line) < 2:
                    logger.debug("_run_paddle: skipping malformed line: %r", line)
                    continue

                bbox_raw  = line[0]
                text_info = line[1]

                # text_info may be (text, confidence) tuple/list or just a string
                if hasattr(text_info, "__len__") and not isinstance(text_info, str):
                    text       = str(text_info[0]).strip()
                    confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
                else:
                    text       = str(text_info).strip()
                    confidence = 0.0

                if not text:
                    continue

                # bbox_raw is either [[x,y],[x,y],[x,y],[x,y]] or [x,y,w,h]
                try:
                    if hasattr(bbox_raw[0], "__len__"):
                        xs   = [float(pt[0]) for pt in bbox_raw]
                        ys   = [float(pt[1]) for pt in bbox_raw]
                        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    else:
                        x, y, w, h = [int(v) for v in bbox_raw]
                        bbox = [x, y, x + w, y + h]
                except Exception:
                    bbox = [0, 0, 0, 0]

                blocks.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                })

            except Exception as e:
                logger.debug("_run_paddle: skipping line due to error: %s — line=%r", e, line)

    return blocks


def _run_tesseract(image: np.ndarray, lang: str = "eng") -> list[dict]:
    """
    Tesseract fallback: run pytesseract on a single numpy image and return
    the same block-dict format as _run_paddle.
    """
    blocks: list[dict] = []
    try:
        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(
            pil_img, lang=lang, output_type=pytesseract.Output.DICT
        )
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf_raw = data["conf"][i]
            confidence = max(0.0, float(conf_raw) / 100.0) if conf_raw != -1 else 0.0
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            blocks.append({
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": [x, y, x + w, y + h],
            })
    except Exception as e:
        logger.error("_run_tesseract failed: %s", e)
    return blocks


# ---------------------------------------------------------------------------
# 3. detect_document_type
# ---------------------------------------------------------------------------

_INVOICE_KEYWORDS = {"gstin", "invoice", "gst", "bill no", "tax invoice", "sgst", "cgst"}
_PRESCRIPTION_KEYWORDS = {"rx", "prescribed", "medicine", "tablet", "capsule", "mg", "dosage", "dr.", "diagnosis"}
_LAB_REPORT_KEYWORDS = {
    "haemoglobin", "hb", "wbc", "platelet", "blood count", "cbc", "crp", "test", "result",
    "normal range", "pathology", "complete blood count", "rbc", "test report", "result value",
    "range value", "differential count", "neutrophils", "lymphocytes", "laboratory",
}


def detect_document_type(raw_text: str) -> str:
    """
    Classify a document as invoice / prescription / lab_report / handwritten_notes
    based on keyword presence in the OCR text.
    Lab report is checked before prescription to avoid misclassification.
    """
    lower = raw_text.lower()

    if any(kw in lower for kw in _INVOICE_KEYWORDS):
        doc_type = "invoice"
    elif any(kw in lower for kw in _LAB_REPORT_KEYWORDS):
        doc_type = "lab_report"
    elif any(kw in lower for kw in _PRESCRIPTION_KEYWORDS):
        doc_type = "prescription"
    else:
        doc_type = "handwritten_notes"

    print(f"Detected document type: {doc_type}")
    logger.info("detect_document_type -> %s", doc_type)
    return doc_type


# ---------------------------------------------------------------------------
# 4. save_uploaded_file
# ---------------------------------------------------------------------------

async def save_uploaded_file(upload_file, destination_folder: str) -> str:
    """
    Persist a FastAPI UploadFile to *destination_folder* and return the
    full path to the saved file.  Creates the folder if it does not exist.
    """
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)

    save_path = dest / upload_file.filename
    contents = await upload_file.read()

    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info("Saved uploaded file to: %s", save_path)
    return str(save_path)
