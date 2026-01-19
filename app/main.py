"""
FactForge - AI-Powered PDF Fact Checker
Main Streamlit Application
"""

import streamlit as st
import os
import sys
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key from Streamlit secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.pdf_parser import extract_text_from_pdf, get_pdf_metadata
from app.claim_extractor import extract_claims
from app.verifier import verify_claims, get_summary_stats

# Page configuration
st.set_page_config(
    page_title="FactForge - PDF Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --verified-color: #10b981;
        --inaccurate-color: #f59e0b;
        --false-color: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #6366f1;
        background: linear-gradient(180deg, #1e293b 0%, #1e1b4b 100%);
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        flex: 1;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .stat-card.verified {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border-color: #10b981;
    }
    
    .stat-card.inaccurate {
        background: linear-gradient(135deg, #78350f 0%, #0f172a 100%);
        border-color: #f59e0b;
    }
    
    .stat-card.false {
        background: linear-gradient(135deg, #7f1d1d 0%, #0f172a 100%);
        border-color: #ef4444;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Claim cards */
    .claim-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .claim-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .claim-card.verified {
        border-left-color: #10b981;
    }
    
    .claim-card.inaccurate {
        border-left-color: #f59e0b;
    }
    
    .claim-card.false {
        border-left-color: #ef4444;
    }
    
    .claim-text {
        font-size: 1.1rem;
        color: #f8fafc;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .claim-meta {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .claim-tag {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .tag-verified {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .tag-inaccurate {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    
    .tag-false {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    .tag-type {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
    }
    
    .explanation-box {
        background: #0f172a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    
    .explanation-title {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }
    
    .explanation-text {
        color: #e2e8f0;
        line-height: 1.6;
    }
    
    .correct-value {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .correct-value-label {
        color: #f59e0b;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .correct-value-text {
        color: #fef3c7;
        font-weight: 600;
    }
    
    .sources-list {
        margin-top: 1rem;
    }
    
    .source-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #334155;
    }
    
    .source-item:last-child {
        border-bottom: none;
    }
    
    .source-link {
        color: #60a5fa;
        text-decoration: none;
        font-size: 0.9rem;
    }
    
    .source-link:hover {
        color: #93c5fd;
        text-decoration: underline;
    }
    
    /* Processing indicator */
    .processing-container {
        text-align: center;
        padding: 3rem;
    }
    
    .stProgress > div > div {
        background-color: #6366f1;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #0f172a;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 FactForge</h1>
        <p>AI-Powered PDF Fact Checker • Verify claims before publishing</p>
    </div>
    """, unsafe_allow_html=True)


def render_stats(stats: dict):
    """Render statistics cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);">
            <div class="stat-number" style="color: #f8fafc;">{stats['total']}</div>
            <div class="stat-label">Total Claims</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card verified">
            <div class="stat-number" style="color: #10b981;">🟢 {stats['verified']}</div>
            <div class="stat-label">Verified</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card inaccurate">
            <div class="stat-number" style="color: #f59e0b;">🟡 {stats['inaccurate']}</div>
            <div class="stat-label">Inaccurate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card false">
            <div class="stat-number" style="color: #ef4444;">🔴 {stats['false']}</div>
            <div class="stat-label">False</div>
        </div>
        """, unsafe_allow_html=True)


def render_claim_card(claim: dict, index: int):
    """Render a single claim card with expandable details."""
    status = claim.get("status", "false").lower()
    status_emoji = {"verified": "🟢", "inaccurate": "🟡", "false": "🔴"}.get(status, "🔴")
    status_class = status if status in ["verified", "inaccurate", "false"] else "false"
    tag_class = f"tag-{status_class}"
    
    # Add special indicators for myths and outdated
    is_myth = claim.get("is_myth", False)
    is_outdated = claim.get("is_outdated", False)
    special_indicator = ""
    if is_myth:
        special_indicator = "🚫 MYTH: "
    elif is_outdated:
        special_indicator = "📅 OUTDATED: "
    
    with st.expander(f"{status_emoji} {special_indicator}{claim.get('claim', 'Unknown claim')[:100]}...", expanded=False):
        # Claim text
        st.markdown(f"""
        <div class="claim-text">{claim.get('claim', 'Unknown claim')}</div>
        """, unsafe_allow_html=True)
        
        # Status and type tags
        claim_type = claim.get("claim_type", "unknown")
        confidence = claim.get("confidence", "low")
        
        tags_html = f"""
        <div class="claim-meta">
            <span class="claim-tag {tag_class}">{status.upper()}</span>
            <span class="claim-tag tag-type">{claim_type}</span>
            <span class="claim-tag tag-type">Confidence: {confidence}</span>
        """
        
        # Add myth/outdated tags
        if is_myth:
            tags_html += '<span class="claim-tag tag-false">🚫 MYTH DETECTED</span>'
        if is_outdated:
            tags_html += '<span class="claim-tag tag-inaccurate">📅 OUTDATED DATA</span>'
        
        tags_html += "</div>"
        st.markdown(tags_html, unsafe_allow_html=True)
        
        # Warning box for myths
        if is_myth:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                <strong style="color: #ef4444;">⚠️ This is a widely circulated myth that has been debunked.</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Warning box for outdated data
        if is_outdated:
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                <strong style="color: #f59e0b;">⏰ This data appears to be outdated. Check below for current values.</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Explanation
        explanation = claim.get("explanation", "No explanation available")
        st.markdown(f"""
        <div class="explanation-box">
            <div class="explanation-title">📋 Reasoning</div>
            <div class="explanation-text">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Correct value (if inaccurate or outdated)
        correct_value = claim.get("correct_value")
        if correct_value:
            st.markdown(f"""
            <div class="correct-value">
                <div class="correct-value-label">✅ Correct/Current Value</div>
                <div class="correct-value-text">{correct_value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sources
        sources = claim.get("sources", [])
        if sources:
            st.markdown("<div class='sources-list'><div class='explanation-title'>🔗 Sources</div>", unsafe_allow_html=True)
            for i, source in enumerate(sources[:3], 1):
                title = source.get("title", f"Source {i}")
                url = source.get("url", "#")
                st.markdown(f"**{i}.** [{title}]({url})", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📖 How it works")
        st.markdown("""
        1. **Upload** a PDF document
        2. **Extract** verifiable claims using AI
        3. **Verify** each claim via web search
        4. **Review** color-coded results
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 What we catch")
        st.markdown("""
        - 🔢 **Statistics** - percentages, numbers
        - 💰 **Financial** - stock prices, market caps
        - 📅 **Dates** - founding years, events
        - 🚫 **Myths** - debunked claims
        - ⏰ **Outdated** - old data presented as current
        - 🔬 **Scientific** - research claims
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **FactForge** uses AI to identify factual claims 
        in PDFs and verify them against current web sources.
        
        Built with Streamlit, LangChain, and Groq.
        """)
    
    # Main content area
    st.markdown("### 📄 Upload PDF Document")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your PDF here",
        type=["pdf"],
        help="Upload a PDF document to check for factual accuracy"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_bytes = BytesIO(uploaded_file.read())
        metadata = get_pdf_metadata(file_bytes)
        file_bytes.seek(0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File", uploaded_file.name[:30] + "..." if len(uploaded_file.name) > 30 else uploaded_file.name)
        with col2:
            st.metric("📑 Pages", metadata.get("pages", "Unknown"))
        with col3:
            st.metric("📦 Size", f"{len(file_bytes.getvalue()) / 1024:.1f} KB")
        
        # Process button
        if st.button("🔍 Analyze Document", use_container_width=True):
            
            # Initialize session state for results
            if "results" not in st.session_state:
                st.session_state.results = None
            
            # Processing pipeline
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Extract text
                status_text.text("📖 Extracting text from PDF...")
                progress_bar.progress(10)
                
                file_bytes.seek(0)
                text = extract_text_from_pdf(file_bytes)
                
                if len(text) < 100:
                    st.error("❌ PDF contains too little text to analyze.")
                    return
                
                progress_bar.progress(20)
                
                # Step 2: Extract claims
                status_text.text("🔎 Identifying factual claims...")
                
                def claim_progress(msg):
                    status_text.text(f"🔎 {msg}")
                
                claims = extract_claims(text, progress_callback=claim_progress)
                
                if not claims:
                    st.warning("⚠️ No verifiable factual claims found in this document.")
                    progress_bar.progress(100)
                    return
                
                progress_bar.progress(40)
                st.info(f"Found **{len(claims)}** factual claims to verify")
                
                # Step 3: Verify claims
                status_text.text("✅ Verifying claims against web sources...")
                
                def verify_progress(msg):
                    status_text.text(f"✅ {msg}")
                    # Update progress based on verification
                    current = progress_bar._current_progress if hasattr(progress_bar, '_current_progress') else 40
                    progress_bar.progress(min(40 + int(60 * (int(msg.split('/')[0].split()[-1]) / len(claims))), 95))
                
                results = verify_claims(claims, progress_callback=verify_progress)
                
                progress_bar.progress(100)
                status_text.text("✨ Analysis complete!")
                
                # Store results
                st.session_state.results = results
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                progress_bar.progress(0)
                return
        
        # Display results
        if st.session_state.get("results"):
            results = st.session_state.results
            
            st.markdown("---")
            st.markdown("### 📊 Verification Results")
            
            # Stats
            stats = get_summary_stats(results)
            render_stats(stats)
            
            st.markdown("---")
            
            # Filter tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                f"📋 All ({stats['total']})",
                f"🟢 Verified ({stats['verified']})",
                f"🟡 Inaccurate ({stats['inaccurate']})",
                f"🔴 False ({stats['false']})"
            ])
            
            with tab1:
                for i, claim in enumerate(results):
                    render_claim_card(claim, i)
            
            with tab2:
                verified = [c for c in results if c.get("status", "").lower() == "verified"]
                if verified:
                    for i, claim in enumerate(verified):
                        render_claim_card(claim, i)
                else:
                    st.info("No verified claims found.")
            
            with tab3:
                inaccurate = [c for c in results if c.get("status", "").lower() == "inaccurate"]
                if inaccurate:
                    for i, claim in enumerate(inaccurate):
                        render_claim_card(claim, i)
                else:
                    st.info("No inaccurate claims found.")
            
            with tab4:
                false_claims = [c for c in results if c.get("status", "").lower() == "false"]
                if false_claims:
                    for i, claim in enumerate(false_claims):
                        render_claim_card(claim, i)
                else:
                    st.info("No false claims found.")
            
            # Export option
            st.markdown("---")
            if st.button("📥 Export Results as JSON"):
                import json
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="factcheck_results.json",
                    mime="application/json"
                )
    
    else:
        # Show placeholder when no file is uploaded
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Drop your PDF here</h3>
            <p style="color: #94a3b8;">or click to browse files</p>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">
                Supported format: PDF • Max recommended size: 10MB
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
