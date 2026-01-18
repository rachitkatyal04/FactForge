# 🔍 FactForge - AI-Powered PDF Fact Checker

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://factforge.streamlit.app)

**FactForge** is an intelligent fact-checking application that verifies factual claims in PDF documents before publishing. It uses AI to extract verifiable claims and cross-references them against live web sources.

![FactForge Screenshot](https://via.placeholder.com/800x400?text=FactForge+Screenshot)

## 🎯 Features

- **📤 Drag-and-Drop PDF Upload** - Simple, intuitive file upload interface
- **🤖 AI-Powered Claim Extraction** - Uses Groq's LLaMA 3.1 to identify verifiable factual claims
- **🌐 Live Web Verification** - Cross-checks claims against current web sources via DuckDuckGo
- **🎨 Color-Coded Results** - Visual classification of claims:
  - 🟢 **Verified** - Matches current authoritative data
  - 🟡 **Inaccurate** - Outdated or numerically incorrect
  - 🔴 **False** - No credible evidence or contradicted by sources
- **📊 Detailed Reasoning** - Shows explanation, corrected values, and source links
- **📥 Export Results** - Download verification results as JSON

## 🏗️ Architecture

```
FactForge/
├── app/
│   ├── __init__.py          # Package initializer
│   ├── main.py              # Streamlit UI application
│   ├── pdf_parser.py        # PDF text extraction using pdfplumber
│   ├── claim_extractor.py   # LLM-based claim extraction
│   ├── verifier.py          # Web search and verification logic
│   └── prompts.py           # LLM prompt templates
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml.example # Example secrets file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **LLM** | Groq API (llama-3.1-8b-instant) |
| **Web Search** | DuckDuckGo (duckduckgo-search) |
| **PDF Parsing** | pdfplumber |
| **LLM Framework** | LangChain |
| **Deployment** | Streamlit Cloud |

## 📋 How It Works

### 1. PDF Upload & Text Extraction
The application accepts PDF files via drag-and-drop. Text is extracted using `pdfplumber`, preserving page structure.

### 2. Claim Extraction (LLM)
The extracted text is processed by Groq's LLaMA 3.1 model to identify:
- Statistics and numerical data
- Specific dates and timelines
- Financial figures
- Technical specifications
- Scientific facts
- Historical events

Opinions, predictions, and vague statements are filtered out.

### 3. Web Verification
Each claim is converted into an optimized search query and verified using:
- DuckDuckGo web search (multiple results per claim)
- LLM analysis comparing claim against search results
- Preference for authoritative sources (government, official reports, reputable news)

### 4. Classification & Results
Claims are classified based on source verification:
- **Verified**: Multiple credible sources confirm the claim
- **Inaccurate**: Claim has errors or is outdated (corrected value provided)
- **False**: No evidence or sources contradict the claim

## 🚀 Local Development

### Prerequisites
- Python 3.9+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/factforge.git
cd factforge
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Option 1: Export directly
export GROQ_API_KEY="your-api-key-here"

# Option 2: Create .streamlit/secrets.toml
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your API key
```

5. **Run the application**
```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`

## ☁️ Deployment to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/factforge.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set the main file path: `app/main.py`
5. Add secrets in the "Advanced settings":
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```
6. Click "Deploy"

### Step 3: Access Your App
Your app will be available at: `https://[your-app-name].streamlit.app`

## 🔐 Security Notes

- **Never commit API keys** - Use environment variables or Streamlit secrets
- **The `.gitignore` excludes** `secrets.toml` and `.env` files
- **API keys entered in the UI** are stored only in session memory

## 📊 Output Format

Verification results follow this JSON structure:

```json
{
  "claim": "The exact factual claim from the document",
  "claim_type": "statistic|date|financial|technical|scientific|historical",
  "entities": ["Company", "Person", "Location"],
  "status": "verified|inaccurate|false",
  "explanation": "Detailed reasoning referencing sources",
  "correct_value": "Corrected value if inaccurate, otherwise null",
  "confidence": "high|medium|low",
  "sources": [
    {
      "title": "Source Title",
      "url": "https://source-url.com",
      "relevance": "How this source supports the conclusion"
    }
  ]
}
```

## ⚠️ Limitations

- **PDF Quality**: Works best with text-based PDFs (not scanned images)
- **Language**: Optimized for English content
- **Rate Limits**: Free Groq API has rate limits; large documents may require patience
- **Search Coverage**: DuckDuckGo may not index all specialized/academic sources
- **Real-time Data**: Some very recent information may not be indexed yet

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com) for fast LLM inference
- [Streamlit](https://streamlit.io) for the amazing framework
- [LangChain](https://langchain.com) for LLM orchestration
- [DuckDuckGo](https://duckduckgo.com) for privacy-focused search

---

**Built with ❤️ for accurate journalism and responsible publishing**
