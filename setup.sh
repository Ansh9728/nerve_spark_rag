#!/bin/bash

# Smart Building RAG System - Clone & Run Setup Script
# This script clones the repository and starts the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

# Default repository URL - replace with your actual repo
DEFAULT_REPO_URL="https://github.com/Ansh9728/nerve_spark_rag.git"

# Function to check if running on Windows
is_windows() {
    [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]
}

# Function to get Python command (prefer explicit python3.11)
get_python_cmd() {
    if is_windows; then
        echo "python"
    else
        # prefer python3.11 if available, otherwise fall back to python3
        if command -v python3.11 &> /dev/null; then
            echo "python3.11"
        else
            echo "python3"
        fi
    fi
}

# Function to activate virtual environment
activate_venv() {
    if is_windows; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
}

# Main setup function
main() {
    print_status "Smart Building RAG System - Clone & Run Setup"
    
    # Step 1: Get repository URL
    print_step "Repository Configuration"
    read -p "Enter repository URL [$DEFAULT_REPO_URL]: " REPO_URL
    REPO_URL=${REPO_URL:-$DEFAULT_REPO_URL}
    
    # Step 2: Get project directory
    read -p "Enter project directory name [smart_building_rag]: " PROJECT_DIR
    PROJECT_DIR=${PROJECT_DIR:-smart_building_rag}
    
    # Step 3: Clone repository
    print_step "Cloning repository..."
    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Directory $PROJECT_DIR already exists. Updating..."
        cd "$PROJECT_DIR"
        git pull origin main || git pull origin master
    else
        git clone "$REPO_URL" "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    
    # Step 4: Check Python
    print_step "Checking Python installation..."
    PYTHON_CMD=$(get_python_cmd)
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python is not installed. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $PYTHON_VERSION"

    # verify minimum version (>=3.11)
    PY_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[0])')
    PY_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[1])')
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
        print_error "Python 3.11 or higher is required. Found $PYTHON_VERSION."
        exit 1
    fi
    
    # Step 5: Create virtual environment
    print_step "Setting up virtual environment..."
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_status "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Step 6: Activate virtual environment and install dependencies
    print_step "Installing dependencies..."
    activate_venv
    pip install --upgrade pip
    pip install -r requirements.txt

    # sanity check: sentence-transformers import
    if ! $PYTHON_CMD -c "import sentence_transformers" &> /dev/null; then
        print_warning "sentence-transformers failed to import; reinstalling specific version"
        pip install --upgrade "sentence-transformers>=5.2.3"
    fi
    
    # Step 7: Check for .env file
    print_step "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Created .env from .env.example - please edit with your database credentials"
        else
            print_warning "No .env file found. Creating default configuration..."
            cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/smart_building_rag

# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=5
RERANK_TOP_N=3

# Scheduler Configuration
SCHEDULER_ENABLED=True
SCHEDULER_INTERVAL_MINUTES=30
EOF
        fi
    fi
    
    # Step 8: Create startup scripts
    print_step "Creating startup scripts..."
    
    # Create start.sh for Unix/Linux/Mac
    if ! is_windows; then
        cat > start.sh << 'EOF'
#!/bin/bash
echo "Starting Smart Building RAG System..."
echo "Starting FastAPI backend..."
gnome-terminal --tab --title="FastAPI" -- bash -c "source venv/bin/activate && python run.py; exec bash" &
sleep 3
echo "Starting Streamlit frontend..."
gnome-terminal --tab --title="Streamlit" -- bash -c "source venv/bin/activate && streamlit run streamlit_app.py; exec bash" &
echo "System started! Access:"
echo "  - Streamlit: http://localhost:8501"
echo "  - FastAPI: http://localhost:8000/docs"
EOF
        chmod +x start.sh
    fi
    
    # Create start.bat for Windows
    if is_windows; then
        cat > start.bat << 'EOF'
@echo off
echo Starting Smart Building RAG System...
echo Starting FastAPI backend...
start cmd /k "venv\Scripts\activate && python run.py"
timeout /t 3
echo Starting Streamlit frontend...
start cmd /k "venv\Scripts\activate && streamlit run streamlit_app.py"
echo System started! Access:
echo   - Streamlit: http://localhost:8501
echo   - FastAPI: http://localhost:8000/docs
pause
EOF
    fi
    
    # Step 9: Create quick-start script
    print_step "Creating quick-start script..."
    
    cat > quick_start.sh << 'EOF'
#!/bin/bash
# Quick start script for Smart Building RAG System

echo "Starting Smart Building RAG System..."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Start FastAPI in background
echo "Starting FastAPI backend on port 8000..."
nohup python run.py > fastapi.log 2>&1 &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 5

# Start Streamlit
echo "Starting Streamlit frontend on port 8501..."
streamlit run streamlit_app.py
EOF
    
    chmod +x quick_start.sh
    
    # Step 10: Final instructions
    print_step "Setup completed successfully!"
    print_status "Project cloned to: $(pwd)"
    print_status "Virtual environment: $(pwd)/venv"
    
    echo ""
    print_status "To start the application:"
    
    if is_windows; then
        print_status "1. Run: start.bat"
    else
        print_status "1. Run: ./start.sh"
    fi
    
    print_status "2. Or manually:"
    print_status "   - Activate venv: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
    print_status "   - Start FastAPI: python run.py"
    print_status "   - Start Streamlit: streamlit run streamlit_app.py"
    
    echo ""
    print_status "Access points:"
    print_status "  - Streamlit UI: http://localhost:8501"
    print_status "  - FastAPI Docs: http://localhost:8000/docs"
    
    # Deactivate virtual environment
    deactivate 2>/dev/null || true
    
    print_status "Ready to use! 🚀"
}

# Run main function
main "$@"
