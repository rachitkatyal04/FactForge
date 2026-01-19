"""
Advanced Claim Extractor Module
Extracts verifiable claims with focus on detecting potential misinformation.
"""

import json
import re
import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import CLAIM_EXTRACTION_SYSTEM_PROMPT, CLAIM_EXTRACTION_USER_PROMPT


def get_llm():
    """Initialize the Groq LLM client - fast model."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
        max_tokens=2048
    )


def chunk_text(text: str, max_chars: int = 5000) -> List[str]:
    """Split text into overlapping chunks for thorough analysis."""
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


def extract_numbers_and_stats(text: str) -> List[Dict[str, Any]]:
    """
    Pre-extract numerical claims that the LLM might miss.
    This ensures we catch ALL statistics.
    """
    claims = []
    
    patterns = [
        # Percentages
        (r'(\d+(?:\.\d+)?%\s*(?:of|increase|decrease|growth|decline|rise|fall)[^.]*\.)', 'statistic'),
        # Money amounts
        (r'(\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion))?[^.]*\.)', 'financial'),
        # Years with context
        (r'((?:in|since|from|founded|established|started)\s*\d{4}[^.]*\.)', 'date'),
        # Large numbers with context
        (r'(\d{1,3}(?:,\d{3})+(?:\s*(?:people|users|customers|employees|downloads))[^.]*\.)', 'statistic'),
    ]
    
    for pattern, claim_type in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 20:  # Filter out very short matches
                claims.append({
                    "claim": match.strip(),
                    "claim_type": claim_type,
                    "entities": [],
                    "search_query": match[:100],
                    "verification_focus": f"Verify the {claim_type} mentioned"
                })
    
    return claims


def extract_json_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Extract claims from LLM response with multiple parsing strategies."""
    
    # Strategy 1: Find ```json block
    if "```json" in response_text:
        try:
            json_str = response_text.split("```json")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return data.get("claims", [])
        except (json.JSONDecodeError, IndexError):
            pass
    
    # Strategy 2: Find ``` block
    if "```" in response_text:
        try:
            json_str = response_text.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return data.get("claims", [])
        except (json.JSONDecodeError, IndexError):
            pass
    
    # Strategy 3: Find JSON object
    try:
        start = response_text.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(response_text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        data = json.loads(response_text[start:i+1])
                        return data.get("claims", [])
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Try whole response
    try:
        data = json.loads(response_text.strip())
        return data.get("claims", [])
    except json.JSONDecodeError:
        pass
    
    return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_claims_from_chunk(llm: ChatGroq, chunk: str) -> List[Dict[str, Any]]:
    """Extract claims from a single text chunk."""
    messages = [
        SystemMessage(content=CLAIM_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=CLAIM_EXTRACTION_USER_PROMPT.format(text=chunk))
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    return extract_json_from_response(response_text)


def normalize_claim(text: str) -> str:
    """Normalize claim text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\d%$.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def claims_are_similar(claim1: str, claim2: str, threshold: float = 0.7) -> bool:
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
    """Remove duplicate or very similar claims."""
    if not claims:
        return []
    
    unique = []
    
    for claim in claims:
        claim_text = claim.get("claim", "").strip()
        
        if not claim_text or len(claim_text) < 15:
            continue
        
        is_duplicate = False
        for existing in unique:
            existing_text = existing.get("claim", "")
            if claims_are_similar(claim_text, existing_text):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(claim)
    
    return unique


def categorize_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure claim has proper categorization."""
    claim_text = claim.get("claim", "").lower()
    
    # Auto-detect claim type if not set
    if not claim.get("claim_type") or claim.get("claim_type") == "unknown":
        if any(word in claim_text for word in ['$', 'billion', 'million', 'revenue', 'profit', 'stock', 'market cap', 'valuation']):
            claim["claim_type"] = "financial"
        elif any(word in claim_text for word in ['%', 'percent', 'ratio', 'rate']):
            claim["claim_type"] = "statistic"
        elif re.search(r'\b(19|20)\d{2}\b', claim_text):
            claim["claim_type"] = "date"
        elif any(word in claim_text for word in ['founded', 'established', 'started', 'launched']):
            claim["claim_type"] = "historical"
        else:
            claim["claim_type"] = "general"
    
    # Generate search query if not present
    if not claim.get("search_query"):
        claim["search_query"] = claim.get("claim", "")[:100]
    
    # Set verification focus if not present
    if not claim.get("verification_focus"):
        claim["verification_focus"] = f"Verify all facts in this {claim['claim_type']} claim"
    
    return claim


def extract_claims(text: str, progress_callback=None) -> List[Dict[str, Any]]:
    """
    Extract all verifiable factual claims from text.
    Uses multiple strategies for thorough extraction.
    """
    llm = get_llm()
    all_claims = []
    
    # Step 1: Pre-extract obvious numerical claims (regex-based)
    if progress_callback:
        progress_callback("Pre-scanning for numerical claims...")
    regex_claims = extract_numbers_and_stats(text)
    all_claims.extend(regex_claims)
    
    # Step 2: LLM-based extraction from chunks
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Analyzing section {i+1}/{len(chunks)}...")
        
        try:
            llm_claims = extract_claims_from_chunk(llm, chunk)
            all_claims.extend(llm_claims)
        except Exception as e:
            print(f"Error extracting from chunk {i+1}: {e}")
            continue
    
    # Step 3: Categorize and validate claims
    categorized_claims = [categorize_claim(c) for c in all_claims]
    
    # Step 4: Deduplicate
    unique_claims = deduplicate_claims(categorized_claims)
    
    # Step 5: Sort by claim type (financial first, then statistics, then others)
    type_priority = {"financial": 0, "statistic": 1, "date": 2, "technical": 3, "scientific": 4, "historical": 5, "general": 6}
    unique_claims.sort(key=lambda x: type_priority.get(x.get("claim_type", "general"), 6))
    
    return unique_claims
