"""
Verifier Module
Verifies factual claims using DuckDuckGo search and LLM analysis.
"""

import json
import re
import os
from typing import Dict, Any, List, Optional
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .prompts import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_PROMPT


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text using multiple strategies.
    """
    # Strategy 1: Try to find ```json ... ``` block
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find ``` ... ``` block
    code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find JSON object pattern { ... }
    json_obj_match = re.search(r'\{[\s\S]*"status"[\s\S]*\}', text)
    if json_obj_match:
        try:
            # Try to find the complete JSON object
            potential_json = json_obj_match.group(0)
            # Balance braces
            brace_count = 0
            end_idx = 0
            for i, char in enumerate(potential_json):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                return json.loads(potential_json[:end_idx])
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Try the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    return None


def parse_text_response(text: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse a non-JSON response to extract verification info.
    """
    text_lower = text.lower()
    
    # Determine status from keywords
    if any(word in text_lower for word in ['verified', 'accurate', 'correct', 'true', 'confirmed', 'matches']):
        if any(word in text_lower for word in ['not verified', 'inaccurate', 'incorrect', 'false', 'outdated', 'wrong']):
            if 'outdated' in text_lower or 'was' in text_lower:
                status = 'inaccurate'
            else:
                status = 'false'
        else:
            status = 'verified'
    elif any(word in text_lower for word in ['outdated', 'old data', 'was correct', 'previously']):
        status = 'inaccurate'
    elif any(word in text_lower for word in ['false', 'incorrect', 'wrong', 'no evidence', 'cannot verify', 'unverified']):
        status = 'false'
    else:
        status = 'false'
    
    # Build sources from search results
    sources = []
    for sr in search_results[:3]:
        sources.append({
            "title": sr.get("title", "Source"),
            "url": sr.get("url", ""),
            "relevance": sr.get("snippet", "")[:200]
        })
    
    return {
        "status": status,
        "explanation": text[:500] if len(text) > 500 else text,
        "correct_value": None,
        "confidence": "medium",
        "sources": sources
    }


def get_llm():
    """Initialize the Groq LLM client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", ""))
            })
            
        return formatted_results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a readable string for the LLM.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string of search results
    """
    if not results:
        return "No search results found."
        
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"""
Source {i}:
Title: {r.get('title', 'N/A')}
URL: {r.get('url', 'N/A')}
Content: {r.get('snippet', 'N/A')}
""")
    
    return "\n".join(formatted)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def verify_claim_with_llm(
    llm: ChatGroq, 
    claim: str, 
    search_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use LLM to verify a claim against search results.
    
    Args:
        llm: The LangChain LLM client
        claim: The claim to verify
        search_results: Web search results
        
    Returns:
        Verification result with status, explanation, and sources
    """
    formatted_results = format_search_results(search_results)
    
    messages = [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=VERIFICATION_USER_PROMPT.format(
            claim=claim,
            search_results=formatted_results
        ))
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Try to extract JSON using multiple strategies
    result = extract_json_from_text(response_text)
    
    if result:
        # Normalize status value
        status = str(result.get("status", "false")).lower().strip()
        if status in ["verified", "true", "correct", "accurate"]:
            result["status"] = "verified"
        elif status in ["inaccurate", "outdated", "partially"]:
            result["status"] = "inaccurate"
        else:
            result["status"] = "false"
        
        # Ensure required fields exist
        result.setdefault("explanation", "Verification completed")
        result.setdefault("correct_value", None)
        result.setdefault("confidence", "medium")
        result.setdefault("sources", [])
        
        return result
    
    # If JSON parsing failed, parse the text response
    return parse_text_response(response_text, search_results)


def verify_single_claim(claim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify a single claim through web search and LLM analysis.
    
    Args:
        claim_data: Claim data including the claim text and search query
        
    Returns:
        Complete verification result
    """
    llm = get_llm()
    
    claim_text = claim_data.get("claim", "")
    search_query = claim_data.get("search_query", claim_text)
    
    # Perform web search
    search_results = search_web(search_query)
    
    # If first search yields no results, try with the claim text directly
    if not search_results and search_query != claim_text:
        search_results = search_web(claim_text)
    
    # Verify with LLM
    verification = verify_claim_with_llm(llm, claim_text, search_results)
    
    # Build final result
    result = {
        "claim": claim_text,
        "claim_type": claim_data.get("claim_type", "unknown"),
        "entities": claim_data.get("entities", []),
        "status": verification.get("status", "false"),
        "explanation": verification.get("explanation", ""),
        "correct_value": verification.get("correct_value"),
        "confidence": verification.get("confidence", "low"),
        "sources": []
    }
    
    # Extract source URLs (limit to 3)
    sources = verification.get("sources", [])
    for source in sources[:3]:
        if isinstance(source, dict) and source.get("url"):
            result["sources"].append({
                "title": source.get("title", "Source"),
                "url": source.get("url", ""),
                "relevance": source.get("relevance", "")
            })
    
    # If no sources from LLM, use search results
    if not result["sources"] and search_results:
        for sr in search_results[:3]:
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
    """
    Verify all extracted claims.
    
    Args:
        claims: List of claims to verify
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of verified claims with results
    """
    results = []
    
    for i, claim in enumerate(claims):
        if progress_callback:
            progress_callback(f"Verifying claim {i+1}/{len(claims)}...")
            
        try:
            result = verify_single_claim(claim)
            results.append(result)
        except Exception as e:
            # If verification fails, mark as unverifiable
            results.append({
                "claim": claim.get("claim", "Unknown claim"),
                "claim_type": claim.get("claim_type", "unknown"),
                "entities": claim.get("entities", []),
                "status": "false",
                "explanation": f"Verification failed: {str(e)}",
                "correct_value": None,
                "confidence": "low",
                "sources": []
            })
    
    return results


def get_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate summary statistics from verification results.
    
    Args:
        results: List of verification results
        
    Returns:
        Dictionary with counts for each status
    """
    stats = {
        "total": len(results),
        "verified": 0,
        "inaccurate": 0,
        "false": 0
    }
    
    for r in results:
        status = r.get("status", "").lower()
        if status == "verified":
            stats["verified"] += 1
        elif status == "inaccurate":
            stats["inaccurate"] += 1
        else:
            stats["false"] += 1
    
    return stats
