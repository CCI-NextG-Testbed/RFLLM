import fitz              # PyMuPDF
import pytesseract
from PIL import Image
import io
import json

# -------------------------
# CONFIG
# -------------------------
PDF_PATH = "PROTOTYPING_AND_DESIGN.pdf"  # change to your file
BOOK_ID = "Juha_Heiskala_&_John_Terry"
BOOK_TITLE = "OFDM Wireless LANs: A Theoretical and Practical Guide"
CHAPTER_TITLE = "RAPID_PROTOTYPING_FOR_WLANs"

CHUNK_SIZE_WORDS = 200   # ~ chunk length
CHUNK_OVERLAP_WORDS = 75

OUTPUT_JSON = "PROTOTYPING_AND_DESIGN.json"


# -------------------------
# OCR: page -> text
# -------------------------
def ocr_page(page, dpi=300):
    """
    Convert a PyMuPDF page to text using Tesseract OCR.
    """
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    # Clean up a bit
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return text

# -------------------------
# Chunking logic
# -------------------------
def chunk_text(text, size=800, overlap=150):
    """
    Chunk text by words with overlap.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append(chunk_text_str)
        if end == len(words):
            break
        start = end - overlap  # slide back by overlap

    return chunks


# -------------------------
# Main: OCR all pages, chunk, build objects
# -------------------------
def process_pdf(path):
    doc = fitz.open(path)
    all_chunks = []

    print(f"Opened PDF: {path}")
    print(f"Total pages: {doc.page_count}")

    for page_index in range(doc.page_count):
        page_num = page_index + 1
        page = doc.load_page(page_index)

        print(f"OCR-ing page {page_num}...")
        text = ocr_page(page)

        if not text:
            print(f"  Page {page_num}: no text extracted (blank or OCR issue).")
            continue

        page_chunks = chunk_text(
            text,
            size=CHUNK_SIZE_WORDS,
            overlap=CHUNK_OVERLAP_WORDS
        )

        print(f"  Page {page_num}: {len(page_chunks)} chunk(s) created.")

        for i, chunk_text_str in enumerate(page_chunks):
            chunk_id = f"{BOOK_ID}_p{page_num}_c{i}"

            chunk_obj = {
                "id": chunk_id,
                "source": BOOK_ID,
                "chapter" : CHAPTER_TITLE,
                "book_title": BOOK_TITLE,
                "page": page_num,
                "chunk_index_on_page": i,
                "text": chunk_text_str,
            }

            all_chunks.append(chunk_obj)

    return all_chunks


if __name__ == "__main__":
    chunks = process_pdf(PDF_PATH)

    print("\n================ SUMMARY ================\n")
    print(f"Total chunks created: {len(chunks)}\n")

    # Save all chunks to JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunks saved to: {OUTPUT_JSON}\n")

    # Print first 3 chunks as formatted JSON, truncating text preview
    print("Sample chunks:\n")
    for c in chunks[:3]:
        preview = c["text"][:300] + ("..." if len(c["text"]) > 300 else "")
        display_obj = {**c, "text": preview}
        print(json.dumps(display_obj, indent=2))
        print("\n---------------------------------------------\n")
