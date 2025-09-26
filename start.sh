
set -e  # Exit on any error

echo "🚀 Starting FastAPI LLM Chat Application..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "main.py not found. Please run this script from the application directory."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p templates static models logs

# Check if templates/index.html exists
if [ ! -f "templates/index.html" ]; then
    print_warning "templates/index.html not found. Please ensure the HTML template is in place."
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade requirements
print_status "Installing/upgrading Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_success "Dependencies installed."

# Check if models directory has any models
MODEL_COUNT=$(find models -name "*.gguf" -o -name "*.bin" -o -name "*.ggml" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    print_warning "No models found in ./models/ directory."
    print_warning "Please add GGUF, BIN, or GGML model files to the models directory."
    print_warning "Example: wget -O models/model.gguf <model_url>"
    print_warning ""
    print_warning "The application will start but won't be functional until models are added."
fi

# Check if llama.cpp server is available in WSL
print_status "Checking llama.cpp server availability..."
if command -v wsl &> /dev/null; then
    if wsl -- test -f "./server"; then
        print_success "llama.cpp server found in WSL."
    else
        print_error "llama.cpp server not found in WSL at ./server"
        print_error "Please compile llama.cpp and ensure the 'server' binary is available."
        print_error "In WSL: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make server"
        exit 1
    fi
else
    print_error "WSL not found. This application requires WSL2 to run llama.cpp."
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the FastAPI application
print_status "Starting FastAPI application..."
print_status "Application will be available at: http://localhost:8000"
print_status "API documentation available at: http://localhost:8000/docs"
print_status ""
print_status "Press Ctrl+C to stop the application."
print_status ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info