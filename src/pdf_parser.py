from pathlib import Path
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF resume.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text, or an empty string if extraction fails.
    """
    path = Path(pdf_path)

    if not path.is_file() or path.suffix.lower() != ".pdf":
        return ""

    try:
        reader = PdfReader(str(path))
        extracted_pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)

        return "\n".join(extracted_pages).strip()

    except Exception as error:
        print(f"Could not read PDF '{path.name}': {error}")
        return ""