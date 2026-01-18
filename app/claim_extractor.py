"""
Claim Extractor Module
Uses Groq LLM via LangChain to extract verifiable factual claims from text.
"""

import json
import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import CLAIM_EXTRACTION_SYSTEM_PROMPT, CLAIM_EXTRACTION_USER_PROMPT


def get_llm():
    """Initialize the Groq LLM client with deterministic settings."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,  # Deterministic output
        max_tokens=4096,
        seed=42  # Fixed seed for reproducibility
    )


def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text: The full text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_chars]]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_claims_from_chunk(llm: ChatGroq, chunk: str) -> List[Dict[str, Any]]:
    """
    Extract claims from a single text chunk using the LLM.
    
    Args:
        llm: The LangChain LLM client
        chunk: Text chunk to analyze
        
    Returns:
        List of extracted claims
    """
    messages = [
        SystemMessage(content=CLAIM_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=CLAIM_EXTRACTION_USER_PROMPT.format(text=chunk))
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Extract JSON from response
    try:
        # Try to find JSON in the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text
            
        data = json.loads(json_str.strip())
        return data.get("claims", [])
        
    except json.JSONDecodeError:
        # Try to extract claims manually if JSON parsing fails
        return []


def extract_claims(text: str, progress_callback=None) -> List[Dict[str, Any]]:
    """
    Extract all verifiable factual claims from the given text.
    
    Args:
        text: The full text to analyze
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of extracted claims with metadata
    """
    llm = get_llm()
    chunks = chunk_text(text)
    all_claims = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Analyzing chunk {i+1}/{len(chunks)}...")
            
        claims = extract_claims_from_chunk(llm, chunk)
        all_claims.extend(claims)
    
    # Deduplicate claims based on similarity
    unique_claims = deduplicate_claims(all_claims)
    
    return unique_claims


def normalize_claim(text: str) -> str:
    """Normalize claim text for comparison."""
    import re
    # Remove extra whitespace, punctuation, and lowercase
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\d%$.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def claims_are_similar(claim1: str, claim2: str, threshold: float = 0.8) -> bool:
    """Check if two claims are similar based on word overlap."""
    words1 = set(normalize_claim(claim1).split())
    words2 = set(normalize_claim(claim2).split())
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    similarity = intersection / union if union > 0 else 0
    return similarity >= threshold


def deduplicate_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate or very similar claims.
    
    Args:
        claims: List of extracted claims
        
    Returns:
        Deduplicated list of claims
    """
    if not claims:
        return []
    
    unique = []
    
    for claim in claims:
        claim_text = claim.get("claim", "").strip()
        
        if not claim_text:
            continue
        
        # Check if similar claim already exists
        is_duplicate = False
        for existing in unique:
            existing_text = existing.get("claim", "")
            if claims_are_similar(claim_text, existing_text):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(claim)
    
    return unique
