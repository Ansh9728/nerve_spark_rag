# Smart Building RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed for smart building management, enabling intelligent document processing and natural language querying capabilities.

## 🏗️ Architecture Overview

This system combines FastAPI backend services with a Streamlit frontend to provide:
- **Document Ingestion**: Multi-format document processing (PDF, DOCX, TXT, MD)
- **Intelligent Retrieval**: Advanced RAG with reranking capabilities
- **Natural Language Querying**: Conversational AI interface for building management
- **Real-time Processing**: Asynchronous document processing with scheduling

## 🚀 Features

### Core Capabilities
- **Multi-format Document Support**: PDF, DOCX, TXT, MD files
- **Advanced RAG Pipeline**: Retrieval with reranking and confidence scoring
- **Real-time Query Interface**: Streamlit-based conversational UI
- **Automated Processing**: Scheduled document ingestion and processing
- **RESTful API**: Complete FastAPI backend with OpenAPI documentation

### Key Components
- **FastAPI Backend**: High-performance async API server
- **Streamlit Frontend**: Interactive web interface
- **PostgreSQL Database**: Persistent storage for documents and metadata
- **Vector Embeddings**: Semantic search capabilities
- **Scheduler**: Automated document processing

## 📁 Project Structure

```
smart_building_rag/
├── app/
│   ├── api/                 # REST API routes
│   ├── core/               # Core configurations and utilities
│   ├── ingestions/         # Document ingestion modules
│   ├── services/          # Business logic services
│   ├── test/              # Test files
│   └── utils/             # Utility functions
├── run.py                 # FastAPI application entry point
├── streamlit_app.py       # Streamlit frontend application
├── requirements.txt       # Python dependencies
└── setup.sh              # One-click setup script
```

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **Database**: PostgreSQL with asyncpg
- **ML/AI**: Sentence Transformers, LangChain
- **Scheduler**: APScheduler
- **File Processing**: PyPDF2, python-docx
- **Vector Storage**: FAISS/ChromaDB compatible

## 🚦 Quick Start

### Prerequisites
- Python 3.8+
- Git

### One-Click Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd smart_building_rag

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Manual Setup
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd smart_building_rag
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

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and settings
   ```

5. **Initialize database**
   ```bash
   # The application will auto-create tables on first run
   ```

6. **Start the services**
   ```bash
   # Terminal 1: Start FastAPI backend
   python run.py

   # Terminal 2: Start Streamlit frontend
   streamlit run streamlit_app.py
   ```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/smart_building_rag

# API Settings
HOST=0.0.0.0
PORT=8000
DEBUG=True

# RAG Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=5
RERANK_TOP_N=3

# Scheduler
SCHEDULER_ENABLED=True
SCHEDULER_INTERVAL_MINUTES=30
```

## 📊 Usage

### 1. Access the Web Interface
- **Streamlit App**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs

### 2. Upload Documents
- Navigate to "Document Management" tab
- Upload PDF, DOCX, TXT, or MD files
- Monitor processing status

### 3. Query Documents
- Use the "Query Interface" tab
- Ask natural language questions
- View confidence scores and source documents

### 4. API Endpoints
- **POST** `/ingest_api/upload-multiple-files` - Upload documents
- **GET** `/retrivel_api/query` - Query documents
- **POST** `/ingest_api/process-documents` - Trigger manual processing

## 🔍 API Documentation

### Document Ingestion
```bash
# Upload files
curl -X POST "http://localhost:8000/ingest_api/upload-multiple-files" \
  -F "files=@document.pdf" \
  -F "files=@document.docx"
```

### Document Query
```bash
# Query documents
curl -X GET "http://localhost:8000/retrivel_api/query?query=What%20is%20IoT%20sensor%20data&top_k=5"
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```bash
   # Check PostgreSQL is running
   sudo systemctl status postgresql
   ```

2. **Port Already in Use**
   ```bash
   # Kill processes on ports 8000 or 8501
   lsof -ti:8000 | xargs kill -9
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --upgrade
   ```
