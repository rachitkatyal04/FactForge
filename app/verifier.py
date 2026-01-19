"""
Advanced Verifier Module
High-accuracy fact verification with myth detection and outdated data catching.
"""

import json
import re
import os
import time
from typing import Dict, Any, List, Optional
try:
    from ddgs import DDGS
except ImportError:
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

# TRUSTED SOURCES - Prioritize these domains
TRUSTED_DOMAINS = [
    # Official & Government
    "gov", ".gov.", "gov.in", "gov.uk", "europa.eu", "un.org", "who.int", "worldbank.org", "imf.org",
    # Major News & Media
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "economist.com", "bloomberg.com", "cnbc.com", "forbes.com", "wsj.com",
    "timesofindia.com", "hindustantimes.com", "ndtv.com", "thehindu.com", "indianexpress.com",
    # Fact-Checkers
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org", "altnews.in",
    # Reference & Academic
    "wikipedia.org", "britannica.com", "encyclopedia.com", "scholar.google.com",
    "sciencedirect.com", "nature.com", "ncbi.nlm.nih.gov", "pubmed.gov",
    # Business & Finance
    "yahoo.com/finance", "finance.yahoo.com", "marketwatch.com", "investing.com",
    "moneycontrol.com", "nseindia.com", "bseindia.com", "sec.gov", "sebi.gov.in",
    # Tech & Science
    "techcrunch.com", "wired.com", "arstechnica.com", "theverge.com", "cnet.com",
    "nasa.gov", "space.com", "scientificamerican.com",
    # Statistics
    "statista.com", "worldometers.info", "ourworldindata.org", "data.gov",
]

# UNTRUSTED SOURCES - Avoid or deprioritize these
BLOCKED_DOMAINS = [
    # Content farms & unreliable
    "medium.com", "quora.com", "answers.com", "ehow.com", "wikihow.com",
    # Known misinformation
    "naturalnews.com", "infowars.com", "breitbart.com", "dailymail.co.uk",
    # Low quality aggregators
    "buzzfeed.com", "huffpost.com",
    # Suspicious TLDs and patterns
    ".cn", ".ru", "baidu.com", "qq.com", "sohu.com", "163.com", "sina.com",
    # SEO spam patterns
    "blogspot", "wordpress.com", "tumblr.com", "weebly.com",
]


def is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted source."""
    url_lower = url.lower()
    for domain in TRUSTED_DOMAINS:
        if domain in url_lower:
            return True
    return False


def is_blocked_source(url: str) -> bool:
    """Check if URL is from a blocked/unreliable source."""
    url_lower = url.lower()
    for domain in BLOCKED_DOMAINS:
        if domain in url_lower:
            return True
    return False


def score_source(result: Dict[str, Any]) -> int:
    """Score a search result based on source reliability."""
    url = result.get("url", "").lower()
    score = 50  # Base score
    
    # Boost for trusted sources
    if is_trusted_source(url):
        score += 100
    
    # Heavy penalty for blocked sources
    if is_blocked_source(url):
        score -= 200
    
    # Boost for specific high-authority domains
    if any(d in url for d in [".gov", "reuters", "apnews", "bbc", "wikipedia", "snopes", "factcheck"]):
        score += 50
    
    # Boost for having snippet content
    if result.get("snippet"):
        score += 10
    
    return score


def filter_and_rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out unreliable sources and rank by trustworthiness."""
    # Remove blocked sources
    filtered = [r for r in results if not is_blocked_source(r.get("url", ""))]
    
    # Sort by trust score (highest first)
    filtered.sort(key=lambda r: score_source(r), reverse=True)
    
    return filtered


def get_llm():
    """Initialize the Groq LLM client - fast model."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
        max_tokens=1024
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


def search_web(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """
    Fast, accurate web search with trusted source filtering.
    Single optimized search - no retries for speed.
    """
    query = query.strip()[:150]
    
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results + 3, region='wt-wt'))
        
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", ""))
            })
        
        # Filter out blocked sources, rank by trust
        return filter_and_rank_results(formatted)[:max_results]
    except Exception as e:
        print(f"Search failed: {e}")
        return []


def search_with_fact_check_sites(claim: str) -> List[Dict[str, Any]]:
    """Fast search on fact-checking websites - single query."""
    try:
        ddgs = DDGS()
        # Combined query for speed
        query = f"{claim[:80]} fact check"
        results = list(ddgs.text(query, max_results=4, region='wt-wt'))
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
    except:
        return []


def search_financial_data(claim: str, entities: List[str]) -> List[Dict[str, Any]]:
    """Fast financial data search - single query."""
    if not entities:
        return []
    try:
        ddgs = DDGS()
        query = f"{entities[0]} stock price market cap 2024 2025"
        results = list(ddgs.text(query, max_results=4, region='wt-wt'))
        return filter_and_rank_results([{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results])
    except:
        return []


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


def search_statistics(claim: str) -> List[Dict[str, Any]]:
    """Fast statistics search - single query."""
    try:
        ddgs = DDGS()
        query = f"{claim[:80]} statistics data"
        results = list(ddgs.text(query, max_results=3, region='wt-wt'))
        return filter_and_rank_results([{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results])
    except:
        return []


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for LLM analysis with source trust indicators."""
    if not results:
        return "[WARNING] NO SEARCH RESULTS FOUND. Mark as FALSE unless you are 100% certain of the fact."
    
    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get('title', 'N/A')
        url = r.get('url', 'N/A')
        snippet = r.get('snippet', 'N/A')
        
        # Add trust indicator
        trust = "[TRUSTED]" if is_trusted_source(url) else "[Standard]"
        
        formatted.append(f"""
=== SOURCE {i} {trust} ===
Title: {title}
URL: {url}
Content: {snippet}
""")
    
    return "\n".join(formatted) + "\n=== END OF SOURCES ==="


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


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=1, max=3))
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
        
        # Ensure all required fields with meaningful defaults
        if not result.get("explanation") or result.get("explanation") == "Verification completed":
            result["explanation"] = "Based on web search results, this claim could not be fully verified with the available sources. Manual verification recommended."
        result.setdefault("correct_value", None)
        result.setdefault("confidence", "medium")
        result.setdefault("is_myth", False)
        result.setdefault("is_outdated", False)
        result.setdefault("sources", [])
        
        return result
    
    return parse_text_response(response_text, search_results)


def verify_single_claim(claim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast + accurate verification: 1-2 smart searches, trusted source filtering.
    """
    llm = get_llm()
    
    claim_text = claim_data.get("claim", "")
    claim_type = claim_data.get("claim_type", "general")
    entities = claim_data.get("entities", [])
    search_query = claim_data.get("search_query", claim_text)
    verification_focus = claim_data.get("verification_focus", "")
    
    # Step 1: Check known myths (instant - no search needed)
    myth_result = check_known_myths(claim_text)
    if myth_result:
        return {
            "claim": claim_text,
            "claim_type": claim_type,
            "entities": entities,
            **myth_result
        }
    
    # Step 2: Single smart search (fast)
    # Use the optimized search query from claim extraction
    search_results = search_web(search_query, max_results=8)
    
    # Only do backup search if we got very few results
    if len(search_results) < 2:
        backup = search_web(claim_text[:80], max_results=5)
        search_results.extend(backup)
    
    # Deduplicate
    seen = set()
    unique_results = []
    for r in search_results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique_results.append(r)
    
    # Step 3: Verify with LLM using search evidence
    verification = verify_claim_with_llm(
        llm, 
        claim_text, 
        unique_results[:8],  # Top 8 results
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
        for sr in search_results:
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
    """Verify all claims with progress updates."""
    results = []
    total = len(claims)
    
    for i, claim in enumerate(claims):
        claim_text = claim.get("claim", "")
        # Show claim number and preview of the claim text (shorter for same line)
        preview = claim_text[:50] + "..." if len(claim_text) > 50 else claim_text
        
        if progress_callback:
            progress_callback(i + 1, total, preview)
        
        try:
            result = verify_single_claim(claim)
            results.append(result)
            # Add delay so progress bar moves visibly (shorter delay for more claims)
            time.sleep(0.3)
        except Exception as e:
            results.append({
                "claim": claim_text,
                "claim_type": claim.get("claim_type", "unknown"),
                "entities": claim.get("entities", []),
                "status": "false",
                "explanation": f"Could not verify: {str(e)}",
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
