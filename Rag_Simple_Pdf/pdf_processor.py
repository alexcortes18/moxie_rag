import hashlib
import os
import re
import PyPDF2
import uuid


class SimplePDFProcessor:
    """Handle PDF processing and chunking."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def readPDF(self, pdf_file):
        """Read PDF and extract the text."""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def compute_file_hash(self, pdf_file):
        """Stable hash of PDF bytes to detect duplicates across sessions."""
        content = pdf_file.getvalue()
        return hashlib.sha256(content).hexdigest()

    def infer_patient_name(self, text, pdf_file):
        """Best-effort patient name inference from PDF text or filename."""
        candidates = []
        patterns = [
            r"Patient\s*Name\s*[:\-]\s*(.+)",
            r"Name\s*[:\-]\s*(.+)",
            r"Patient\s*[:\-]\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                raw = raw.split("\n")[0].strip()
                raw = re.sub(r"\s{2,}", " ", raw)
                if 3 <= len(raw) <= 80:
                    candidates.append(raw)
        if candidates:
            return candidates[0]

        # Fallback to filename without extension
        base = os.path.splitext(pdf_file.name)[0]
        base = base.replace("_", " ").replace("-", " ").strip()
        return base or "Unknown"

    def create_chunks(self, text, pdf_file, patient_name=None, file_hash=None):
        """Split text into chunks."""
        chunks = []
        start = 0

        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size

            # If not at the start, include overlap
            if start > 0:
                start = start - self.chunk_overlap

            # Get chunk
            chunk = text[start:end]

            # Try to end the chunk at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "source": pdf_file.name,
                        "patient_name": patient_name or "Unknown",
                        "file_hash": file_hash,
                    },
                }
            )
            start = end
        return chunks
