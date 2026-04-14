import logging

logger = logging.getLogger(__name__)


def extract_fields(raw_text: str, doc_type: str) -> dict:
    """
    Return all OCR text as readable grouped lines and a full paragraph.
    Short single-word fragments are merged into the previous line to avoid
    the single-word-per-row problem from OCR block output.
    """
    lines = raw_text.splitlines()
    meaningful_lines = [l.strip() for l in lines if len(l.strip()) > 2]

    # Build full paragraph for chat context
    full_paragraph = " ".join(meaningful_lines)

    # Group short fragments (<4 chars) into previous line
    grouped_lines = []
    for line in meaningful_lines:
        if grouped_lines and len(line) < 4:
            grouped_lines[-1] = grouped_lines[-1] + " " + line
        else:
            grouped_lines.append(line)

    result = {
        "doc_type": doc_type,
        "all_lines": grouped_lines,
        "total_lines": len(grouped_lines),
        "full_text": full_paragraph,
        "confidence_score": 1.0 if grouped_lines else 0.0,
    }
    logger.info(
        "extract_fields: doc_type=%r total_lines=%d words=%d",
        doc_type, len(grouped_lines), len(full_paragraph.split()),
    )
    return result
