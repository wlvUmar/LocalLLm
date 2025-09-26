# FastAPI LLM Chat Application

A modern, responsive web interface for chatting with local Large Language Models (LLMs) using llama.cpp. Features real-time streaming responses, voice input, model switching, and a clean, intuitive UI. **Now enhanced with database integration, tool system, and context management!**

![Application Screenshot](https://via.placeholder.com/800x500?text=FastAPI+LLM+Chat+Interface)

## 🆕 Enhanced Features

### 🧠 **NEW: Context Management & Memory**
- **Persistent Conversation Storage**: PostgreSQL database for all conversations
- **Vector Search**: ChromaDB for semantic search of conversation history
- **Tool System**: LLM can use tools to query database and maintain context
- **Conversation State Tracking**: Remember conversation steps and context
- **Short-term Memory**: LLM maintains context throughout conversations
- **Long-term Memory**: Vector database for finding relevant past conversations

### 🔧 **NEW: Tool System**
The LLM can now use tools to:
- **search_history(query)**: Search conversation history for relevant information
- **get_context()**: Get current conversation context and step
- **update_step(step, summary)**: Update conversation step and summary

### 📊 **NEW: Database Integration**
- **PostgreSQL**: Persistent storage for all conversations
- **ChromaDB**: Vector database for semantic search
- **Alembic**: Database migrations and schema management
- **Async Support**: Full async/await database operations

## Features

### 🚀 Core Features
- **Modern Chat Interface**: Clean, responsive design with real-time streaming
- **Voice Input**: Web Speech API integration for hands-free interaction
- **Model Management**: Easy switching between different LLM models
- **Real-time Status**: Live server status and model information
- **Stream Responses**: Real-time response streaming for better UX
- **Mobile Responsive**: Works great on desktop, tablet, and mobile devices

### 🔧 Technical Features
- **FastAPI Backend**: High-performance async Python web framework
- **Jinja2 Templates**: Server-side rendering with modern HTML/CSS/JS
- **WSL2 Integration**: Seamless integration with llama.cpp in WSL2
- **Process Management**: Automatic LLM server lifecycle management
- **Error Handling**: Comprehensive error handling and user feedback
- **Logging**: Detailed logging for debugging and monitoring

## Prerequisites

### Required Software
- **Python 3.8+** (tested with Python 3.11)
- **PostgreSQL** (for conversation storage)
- **WSL2** (Windows Subsystem for Linux 2)
- **llama.cpp** compiled in WSL2

### Database Setup
The application now requires PostgreSQL for conversation storage. You can set it up using Docker:

```bash
# Start PostgreSQL with Docker
docker run -d \
  --name postgres-llm \
  -e POSTGRES_PASSWORD=getout04 \
  -e POSTGRES_DB=postgres \
  -p 5433:5432 \
  postgres:15
```

### WSL2 Setup
1. Install WSL2 on Windows:
   ```bash
   wsl --install
   ```

2. Install build dependencies in WSL2:
   ```bash
   sudo apt update
   sudo apt install build-essential git cmake
   ```

3. Clone and compile llama.cpp:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make server
   ```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd fastapi-llm-chat
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Database
```bash
# Run the database setup script
python setup_db.py
```

### 4. Create Directory Structure
```bash
mkdir -p models templates static logs chroma_data
```

### 5. Add HTML Template
Save the provided HTML template as `templates/index.html`.

### 6. Download Models
Download GGUF model files to the `models/` directory:
```bash
# Example: Download a small model
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Usage

### Quick Start
1. **Start the application**:
   ```bash
   # Make startup script executable
   chmod +x start.sh
   
   # Run the startup script
   ./start.sh
   ```

2. **Access the interface**:
   - Open your browser to `http://localhost:8000`
   - The chat interface will load automatically
   - Select a model from the dropdown and start chatting!

### Manual Start
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using the Interface

#### Basic Chat
1. Type your message in the input field
2. Press Enter or click the send button
3. Watch the AI response stream in real-time

#### Voice Input
1. Click the microphone button
2. Speak your message clearly
3. The text will appear in the input field
4. Click send to submit

#### Model Switching
1. Select a different model from the dropdown
2. Click the "Switch" button
3. Wait for the model to load (status indicator will show progress)
4. Start chatting with the new model

## API Endpoints

### Web Interface
- `GET /` - Main chat interface

### API Endpoints
- `GET /api/status` - Get server status and current model
- `GET /api/models` - List available models
- `POST /api/chat` - Send chat message
- `POST /api/switch-model` - Switch to different model

### Example API Usage
```python
import requests

# Check status
response = requests.get("http://localhost:8000/api/status")
print(response.json())

# Send chat message
response = requests.post("http://localhost:8000/api/chat", json={
    "message": "Hello, how are you?",
    "stream": False
})
print(response.json())

# Switch model
response = requests.post("http://localhost:8000/api/switch-model", json={
    "model": "your-model-name.gguf"
})
print(response.json())
```

## Configuration

### Environment Variables
Create a `.env` file to customize settings:
```env
# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# LLM settings
MODELS_DIR=models
LLM_SERVER_PORT=8080
LLM_STARTUP_TIMEOUT=30

# Default parameters
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
DEFAULT_MAX_TOKENS=512

# Logging
LOG_LEVEL=INFO
```

### Model Directory Structure
```
models/
├── model1.gguf
├── model2.gguf
└── model3.bin
```

## Troubleshooting

### Common Issues

#### "No models found"
- Ensure model files are in the `models/` directory
- Check that models have supported extensions (.gguf, .bin, .ggml)
- Verify file permissions

#### "LLM server not ready"
- Check that llama.cpp server binary exists in WSL2
- Verify WSL2 is running: `wsl --list --running`
- Check server logs for errors

#### "Failed to switch model"
- Ensure the model file exists and is readable
- Check available disk space
- Verify the model file isn't corrupted

#### Voice input not working
- Ensure you're using HTTPS or localhost
- Check browser permissions for microphone access
- Verify Web Speech API support in your browser

### Debugging

#### Enable Debug Mode
```bash
export DEBUG=true
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

#### Check Logs
```bash
# View application logs
tail -f logs/app.log

# Check WSL2 process
wsl -- ps aux | grep server
```

## Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t fastapi-llm-chat .

# Run the container
docker run -d \
  --name llm-chat \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fastapi-llm-chat
```

### Docker Compose
```yaml
version: '3.8'
services:
  llm-chat:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
```

## Performance Optimization

### System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB+ RAM, 8+ CPU cores
- **Storage**: 10GB+ free space for models

### Model Selection
- **Small models** (1-3B parameters): Fast, good for simple tasks
- **Medium models** (7-13B parameters): Balanced performance and quality
- **Large models** (30B+ parameters): Best quality, requires more resources

### Performance Tips
1. Use quantized models (Q4, Q5, Q8) for better performance
2. Adjust context size based on your needs
3. Monitor system resources during use
4. Use SSD storage for faster model loading

## Development

### Project Structure
```
fastapi-llm-chat/
├── main.py              # Application entry point
├── app/                 # Application package
│   ├── __init__.py      # Package initialization
│   ├── main.py          # FastAPI app creation
│   ├── models.py        # Pydantic data models
│   ├── routes.py        # API routes and endpoints
│   ├── events.py        # Startup/shutdown events
│   └── llm_manager.py   # LLM server management
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── start.sh            # Startup script
├── Dockerfile          # Docker configuration
├── README.md           # This file
├── templates/
│   └── index.html      # Chat interface template
├── static/             # Static files (CSS, JS, images)
├── models/             # LLM model files
└── logs/               # Application logs
```

### Adding Features
1. **New API endpoints**: Add to `app/routes.py`
2. **UI changes**: Modify `templates/index.html`
3. **Configuration**: Update `config.py`
4. **Styling**: Add CSS to the HTML template or static files

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Bootstrap](https://getbootstrap.com/) - CSS framework for responsive design

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and system information

---

**Happy chatting with your local LLMs! 🤖💬**

