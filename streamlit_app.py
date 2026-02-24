import streamlit as st
import os
from app.core.config import settings
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Page configuration
st.set_page_config(
    page_title="Smart Building RAG System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
# API_BASE_URL = f"http://{settings.HOST}:{settings.PORT}"
# API_BASE_URL = settings.API_BASE_URL
API_BASE_URL = "http://localhost:7777"

class RAGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    def post(self, endpoint: str, files: Dict = None, data: Dict = None) -> Dict:
        try:
            if files:
                response = requests.post(f"{self.base_url}{endpoint}", files=files)
            else:
                response = requests.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {}

# Initialize client
client = RAGClient(API_BASE_URL)

# Sidebar navigation
st.sidebar.title("🏢 Smart Building RAG")
page = st.sidebar.selectbox(
    "Choose a section",
    ["Query Interface", "Document Management",]
)


# Query Interface Page
if page == "Query Interface":
    st.markdown('<h1 class="main-header">Smart Building Query Interface</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Ask Your Question")
        query = st.text_area("Enter your question:", placeholder="e.g., What is key requirement of IoT sensor data?", height=100)

        col_params1, col_params2 = st.columns(2)
        with col_params1:
            top_k = st.slider("Top K Documents", 1, 15, 5)
            rerank_top_n = st.slider("Rerank Top N", 1, 8, 3)
        with col_params2:
            include_sources = st.checkbox("Include Sources", value=True)

        if st.button("🚀 Submit Query", type="primary"):
            if query:
                with st.spinner("Processing your query..."):
                    params = {
                        "query": query,
                        "top_k": top_k,
                        "rerank_top_n": rerank_top_n,
                        "include_sources": include_sources
                    }
                    response = client.get("/retrivel_api/query", params=params)

                    if response:
                        st.success("✅ Query processed successfully!")

                        # --- Answer ---
                        st.markdown("### 💡 Answer")
                        st.info(response.get("answer", "⚠️ No answer received"))

                        # --- Metrics ---
                        st.markdown("### 📊 Metrics Overview")

                        metrics = response.get("metrics", {})
                        perf = response.get("performance", {})

                        # Combine metrics & performance
                        all_metrics = {**metrics, **perf}
                        df_metrics = pd.DataFrame(list(all_metrics.items()), columns=["Metric", "Value"])
                        st.dataframe(df_metrics, use_container_width=True)

                        # --- Visualizations ---
                        col_v1, col_v2 = st.columns(2)

                        with col_v1:
                            # Time breakdown pie
                            times = {
                                "Retrieval": perf.get("retrieval_time_ms", 0),
                                "Generation": perf.get("generation_time_ms", 0),
                                "Other": perf.get("total_time_ms", 0) - (
                                    perf.get("retrieval_time_ms", 0) + perf.get("generation_time_ms", 0)
                                )
                            }
                            fig_pie = px.pie(names=list(times.keys()), values=list(times.values()), title="⏱ Time Breakdown")
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col_v2:
                            # Confidence gauge
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=response.get("confidence_score", 0),
                                title={"text": "Confidence Score"},
                                gauge={
                                    "axis": {"range": [0, 1]},
                                    "bar": {"color": "green"},
                                    "steps": [
                                        {"range": [0, 0.3], "color": "red"},
                                        {"range": [0.3, 0.7], "color": "yellow"},
                                        {"range": [0.7, 1], "color": "lightgreen"}
                                    ]
                                }
                            ))
                            st.plotly_chart(fig_gauge, use_container_width=True)

                        # --- Sources ---
                        if include_sources and response.get("sources"):
                            st.markdown("### 📚 Sources")
                            for idx, source in enumerate(response["sources"]):
                                metadata = source.get("metadata", {})
                                with st.expander(f"Source {idx + 1} ({metadata.get('filename', 'Unknown')})"):
                                    file_type = metadata.get("file_type", "Unknown").upper()
                                    st.markdown(f"**Type:** `{file_type}` | **Score:** {source.get('score', 0):.3f}")
                                    st.write(f"**Content Preview:** {source.get('content', '')[:300]}...")
            else:
                st.warning("⚠️ Please enter a question before submitting.")
    

# Document Management Page
elif page == "Document Management":
    st.markdown('<h1 class="main-header">Document Management</h1>', unsafe_allow_html=True)
    
    tab1, tab2, = st.tabs(["📁 Upload Documents", "🔄 Process Documents"])
    
    with tab1:
        st.subheader("Upload Multiple Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("📤 Upload Files"):
                files = []
                for file in uploaded_files:
                    files.append(("files", (file.name, file.getvalue(), file.type)))
                
                with st.spinner("Uploading and processing documents..."):
                    response = client.post("/ingest_api/upload-multiple-files", files=files)
                    
                    if response:
                        st.success("Files uploaded successfully!")
                        st.json(response)
    
    
    with tab2:
        st.subheader("Process Documents")
        
        if st.button("🔄 Trigger Manual Processing"):
            with st.spinner("Processing documents..."):
                response = client.post("/ingest_api/process-documents")
                
                if response:
                    st.success("Document processing completed!")
                    st.json(response)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Smart Building RAG System** v1.0")
st.sidebar.markdown("Built with Streamlit & FastAPI")
