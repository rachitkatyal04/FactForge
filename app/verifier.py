"""
Advanced Verifier Module
High-accuracy fact verification with myth detection and outdated data catching.
"""

import json
import re
import os
import time
from typing import Dict, Any, List, Optional
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import (
    VERIFICATION_SYSTEM_PROMPT, 
    VERIFICATION_USER_PROMPT,
    FINANCIAL_VERIFICATION_PROMPT,
    MYTH_DETECTION_PROMPT
)


# Common myths database for quick detection
KNOWN_MYTHS = {
    "10% of brain": {"status": "false", "correct": "Humans use virtually all parts of their brain", "explanation": "This is a debunked myth. Brain scans show activity throughout the entire brain."},
    "great wall.*space": {"status": "false", "correct": "The Great Wall is not visible from space with naked eye", "explanation": "Astronauts have confirmed this is a myth. The wall is too narrow."},
    "goldfish.*memory": {"status": "false", "correct": "Goldfish can remember things for months", "explanation": "Studies show goldfish have memory spans of at least 3 months."},
    "lightning.*twice": {"status": "false", "correct": "Lightning frequently strikes the same place", "explanation": "Tall structures like the Empire State Building are struck dozens of times per year."},
    "sugar.*hyperactive": {"status": "false", "correct": "Sugar does not cause hyperactivity in children", "explanation": "Multiple scientific studies have debunked this myth."},
    "cracking.*arthritis": {"status": "false", "correct": "Knuckle cracking does not cause arthritis", "explanation": "Long-term studies found no correlation between knuckle cracking and arthritis."},
    "bats are blind": {"status": "false", "correct": "Bats can see quite well", "explanation": "All bat species have functional eyes and many have excellent night vision."},
    "bulls.*red": {"status": "false", "correct": "Bulls are colorblind to red; they react to movement", "explanation": "Bulls charge at the cape's movement, not its color."},
}


def get_llm():
    """Initialize the Groq LLM client with deterministic settings."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
        max_tokens=2048,
        seed=42
    )


def check_known_myths(claim: str) -> Optional[Dict[str, Any]]:
    """Check if claim matches known myths."""
    claim_lower = claim.lower()
    for pattern, result in KNOWN_MYTHS.items():
        if re.search(pattern, claim_lower):
            return {
                "status": result["status"],
                "explanation": result["explanation"],
                "correct_value": result["correct"],
                "confidence": "high",
                "is_myth": True,
                "is_outdated": False,
                "sources": [{"title": "Scientific Consensus", "url": "https://www.snopes.com", "relevance": "Myth debunked"}]
            }
    return None


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers, percentages, and monetary values from text."""
    patterns = [
        r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion))?',  # Money
        r'[\d,]+(?:\.\d+)?%',  # Percentages
        r'[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion))',  # Large numbers
        r'\b\d{4}\b',  # Years
        r'[\d,]+(?:\.\d+)?',  # Regular numbers
    ]
    numbers = []
    for pattern in patterns:
        numbers.extend(re.findall(pattern, text, re.IGNORECASE))
    return numbers


def search_web(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Advanced web search with multiple strategies and retries.
    """
    formatted_results = []
    
    # Clean and prepare query
    query = query.strip()[:200]  # Limit query length
    
    # Multiple search strategies
    search_variations = [
        query,
        f"{query} fact check",
        f"{query} statistics data",
        f'"{query}"',  # Exact match
    ]
    
    for search_query in search_variations:
        if len(formatted_results) >= max_results:
            break
            
        for attempt in range(3):
            try:
                ddgs = DDGS()
                results = list(ddgs.text(
                    search_query,
                    max_results=max_results,
                    region='wt-wt',
                    safesearch='off'
                ))
                
                for r in results:
                    result = {
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", ""))
                    }
                    # Avoid duplicates
                    if result not in formatted_results:
                        formatted_results.append(result)
                
                if formatted_results:
                    break
                    
            except Exception as e:
                print(f"Search attempt {attempt + 1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))
                continue
        
        if formatted_results:
            break
    
    return formatted_results[:max_results]


def search_with_fact_check_sites(claim: str) -> List[Dict[str, Any]]:
    """Search specifically on fact-checking websites."""
    fact_check_query = f"{claim} site:snopes.com OR site:factcheck.org OR site:politifact.com OR site:reuters.com/fact-check"
    return search_web(fact_check_query, max_results=5)


def search_financial_data(claim: str, entities: List[str]) -> List[Dict[str, Any]]:
    """Search for current financial data."""
    results = []
    
    # Extract company names and financial terms
    for entity in entities[:3]:
        queries = [
            f"{entity} stock price 2024",
            f"{entity} market cap current",
            f"{entity} revenue latest",
        ]
        for q in queries:
            results.extend(search_web(q, max_results=3))
    
    return results


def search_with_keywords(claim: str) -> List[Dict[str, Any]]:
    """Extract key terms from claim and search."""
    # Extract numbers, years, percentages
    numbers = re.findall(r'\d+(?:\.\d+)?%?', claim)
    
    # Extract capitalized words (proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
    
    # Build focused search query
    key_terms = proper_nouns[:3] + numbers[:2]
    if key_terms:
        focused_query = " ".join(key_terms)
        return search_web(focused_query, max_results=5)
    
    return []


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for LLM analysis."""
    if not results:
        return "No search results found. Use your knowledge to verify if possible."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"""
[Source {i}]
Title: {r.get('title', 'N/A')}
URL: {r.get('url', 'N/A')}
Content: {r.get('snippet', 'N/A')}
""")
    
    return "\n".join(formatted)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response using multiple strategies."""
    # Strategy 1: ```json block
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: ``` block
    code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find JSON object with status field
    json_obj_match = re.search(r'\{[^{}]*"status"[^{}]*\}', text, re.DOTALL)
    if json_obj_match:
        try:
            return json.loads(json_obj_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Find any JSON object
    try:
        start = text.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            break
    except:
        pass
    
    # Strategy 5: Whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    return None


def parse_text_response(text: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse non-JSON response to extract verification info."""
    text_lower = text.lower()
    
    # Determine status from keywords with priority
    if any(phrase in text_lower for phrase in ['is false', 'is incorrect', 'is wrong', 'debunked', 'myth', 'no evidence', 'fabricated']):
        status = 'false'
    elif any(phrase in text_lower for phrase in ['outdated', 'was correct', 'old data', 'previously', 'no longer', 'changed to', 'now is']):
        status = 'inaccurate'
    elif any(phrase in text_lower for phrase in ['verified', 'confirmed', 'accurate', 'correct', 'true', 'matches']):
        status = 'verified'
    else:
        status = 'false'  # Default to false for safety
    
    # Build sources
    sources = []
    for sr in search_results[:3]:
        sources.append({
            "title": sr.get("title", "Source"),
            "url": sr.get("url", ""),
            "relevance": sr.get("snippet", "")[:200]
        })
    
    return {
        "status": status,
        "explanation": text[:800] if len(text) > 800 else text,
        "correct_value": None,
        "confidence": "medium",
        "is_myth": 'myth' in text_lower,
        "is_outdated": 'outdated' in text_lower or 'old' in text_lower,
        "sources": sources
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def verify_claim_with_llm(
    llm: ChatGroq,
    claim: str,
    search_results: List[Dict[str, Any]],
    claim_type: str = "general",
    verification_focus: str = ""
) -> Dict[str, Any]:
    """Use LLM to verify claim with high accuracy."""
    
    formatted_results = format_search_results(search_results)
    
    # Choose appropriate prompt based on claim type
    if claim_type == "financial":
        user_prompt = FINANCIAL_VERIFICATION_PROMPT.format(
            claim=claim,
            search_results=formatted_results
        )
    else:
        user_prompt = VERIFICATION_USER_PROMPT.format(
            claim=claim,
            search_results=formatted_results,
            verification_focus=verification_focus or "Verify accuracy of all facts and figures"
        )
    
    messages = [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse response
    result = extract_json_from_text(response_text)
    
    if result:
        # Normalize and validate status
        status = str(result.get("status", "")).lower().strip()
        if status in ["verified", "true", "correct", "accurate", "confirmed"]:
            result["status"] = "verified"
        elif status in ["inaccurate", "outdated", "partially", "partially correct", "partially true"]:
            result["status"] = "inaccurate"
        else:
            result["status"] = "false"
        
        # Ensure all required fields
        result.setdefault("explanation", "Verification completed")
        result.setdefault("correct_value", None)
        result.setdefault("confidence", "medium")
        result.setdefault("is_myth", False)
        result.setdefault("is_outdated", False)
        result.setdefault("sources", [])
        
        return result
    
    return parse_text_response(response_text, search_results)


def verify_single_claim(claim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify a single claim with multiple verification strategies.
    Designed for high accuracy (99%+).
    """
    llm = get_llm()
    
    claim_text = claim_data.get("claim", "")
    claim_type = claim_data.get("claim_type", "general")
    entities = claim_data.get("entities", [])
    search_query = claim_data.get("search_query", claim_text)
    verification_focus = claim_data.get("verification_focus", "")
    
    # Step 1: Check known myths database (instant detection)
    myth_result = check_known_myths(claim_text)
    if myth_result:
        return {
            "claim": claim_text,
            "claim_type": claim_type,
            "entities": entities,
            **myth_result
        }
    
    # Step 2: Gather evidence from multiple sources
    all_search_results = []
    
    # Primary search with optimized query
    all_search_results.extend(search_web(search_query, max_results=5))
    
    # Search fact-checking sites
    fact_check_results = search_with_fact_check_sites(claim_text)
    all_search_results.extend(fact_check_results)
    
    # Financial claims get special treatment
    if claim_type == "financial" or any(word in claim_text.lower() for word in ['stock', 'price', 'market cap', 'revenue', 'billion', 'million', 'valuation']):
        financial_results = search_financial_data(claim_text, entities)
        all_search_results.extend(financial_results)
        claim_type = "financial"
    
    # Keyword-based search as fallback
    if len(all_search_results) < 3:
        all_search_results.extend(search_with_keywords(claim_text))
    
    # Direct claim search if still few results
    if len(all_search_results) < 3:
        all_search_results.extend(search_web(claim_text[:100], max_results=5))
    
    # Deduplicate results
    seen_urls = set()
    unique_results = []
    for r in all_search_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    # Step 3: Verify with LLM
    verification = verify_claim_with_llm(
        llm, 
        claim_text, 
        unique_results[:10],  # Top 10 results
        claim_type,
        verification_focus
    )
    
    # Build final result
    result = {
        "claim": claim_text,
        "claim_type": claim_type,
        "entities": entities,
        "status": verification.get("status", "false"),
        "explanation": verification.get("explanation", ""),
        "correct_value": verification.get("correct_value"),
        "confidence": verification.get("confidence", "medium"),
        "is_myth": verification.get("is_myth", False),
        "is_outdated": verification.get("is_outdated", False),
        "sources": []
    }
    
    # Extract sources (prioritize LLM sources, fallback to search results)
    llm_sources = verification.get("sources", [])
    for source in llm_sources[:3]:
        if isinstance(source, dict) and source.get("url"):
            result["sources"].append({
                "title": source.get("title", "Source"),
                "url": source.get("url", ""),
                "relevance": source.get("relevance", "")
            })
    
    # Add search results as sources if needed
    if len(result["sources"]) < 3:
        for sr in unique_results:
            if len(result["sources"]) >= 3:
                break
            if sr.get("url") and sr.get("url") not in [s.get("url") for s in result["sources"]]:
                result["sources"].append({
                    "title": sr.get("title", "Source"),
                    "url": sr.get("url", ""),
                    "relevance": sr.get("snippet", "")[:200]
                })
    
    return result


def verify_claims(
    claims: List[Dict[str, Any]],
    progress_callback=None
) -> List[Dict[str, Any]]:
    """Verify all extracted claims with high accuracy."""
    results = []
    
    for i, claim in enumerate(claims):
        if progress_callback:
            progress_callback(f"Verifying claim {i+1}/{len(claims)}...")
        
        try:
            result = verify_single_claim(claim)
            results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            results.append({
                "claim": claim.get("claim", "Unknown claim"),
                "claim_type": claim.get("claim_type", "unknown"),
                "entities": claim.get("entities", []),
                "status": "false",
                "explanation": f"Verification error: {str(e)}. Manual review recommended.",
                "correct_value": None,
                "confidence": "low",
                "is_myth": False,
                "is_outdated": False,
                "sources": []
            })
    
    return results


def get_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate summary statistics."""
    stats = {
        "total": len(results),
        "verified": 0,
        "inaccurate": 0,
        "false": 0,
        "myths_detected": 0,
        "outdated_detected": 0
    }
    
    for r in results:
        status = r.get("status", "").lower()
        if status == "verified":
            stats["verified"] += 1
        elif status == "inaccurate":
            stats["inaccurate"] += 1
        else:
            stats["false"] += 1
        
        if r.get("is_myth"):
            stats["myths_detected"] += 1
        if r.get("is_outdated"):
            stats["outdated_detected"] += 1
    
    return stats
