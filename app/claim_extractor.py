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
    """Initialize the Groq LLM client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096
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
        
    seen_claims = set()
    unique = []
    
    for claim in claims:
        claim_text = claim.get("claim", "").lower().strip()
        
        # Simple deduplication based on exact match
        if claim_text and claim_text not in seen_claims:
            seen_claims.add(claim_text)
            unique.append(claim)
    
    return unique
