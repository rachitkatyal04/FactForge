"""
PDF Parser Module
Extracts text content from uploaded PDF files using pdfplumber.
"""

import pdfplumber
from io import BytesIO
from typing import Optional


def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """
    Extract all text content from a PDF file.
    
    Args:
        pdf_file: A file-like object containing the PDF data
        
    Returns:
        Extracted text as a single string
    """
    text_content = []
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num} ---\n{page_text}")
                    
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    if not text_content:
        raise ValueError("No text content found in the PDF")
        
    return "\n\n".join(text_content)


def get_pdf_metadata(pdf_file: BytesIO) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_file: A file-like object containing the PDF data
        
    Returns:
        Dictionary containing PDF metadata
    """
    metadata = {
        "pages": 0,
        "title": None,
        "author": None
    }
    
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            metadata["pages"] = len(pdf.pages)
            if pdf.metadata:
                metadata["title"] = pdf.metadata.get("Title")
                metadata["author"] = pdf.metadata.get("Author")
    except Exception:
        pass
        
    return metadata
