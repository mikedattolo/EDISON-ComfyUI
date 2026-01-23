# EDISON - Enhanced Distributed Intelligence System for Offline Networks

## Project Description

EDISON is a production-ready, fully offline AI platform that brings enterprise-level language model capabilities to local infrastructure. Built for privacy-conscious organizations and power users, EDISON provides ChatGPT/Claude-level functionality without any cloud dependencies or API costs.

### Key Highlights

- **100% Private & Offline** - All AI processing happens locally on your hardware
- **Multi-Modal Capabilities** - Text generation, vision understanding, and image creation
- **Production Ready** - Systemd services, comprehensive logging, automatic crash recovery
- **Modern Web Interface** - Clean, responsive UI inspired by leading AI chat platforms
- **RAG Memory System** - Long-term context retention using vector embeddings
- **Multi-GPU Support** - Efficient tensor splitting across multiple NVIDIA GPUs
- **Zero API Costs** - One-time hardware investment, unlimited usage

### Technology Stack

- **Backend**: FastAPI + llama-cpp-python with CUDA acceleration
- **LLM Models**: Qwen 2.5 (14B/72B) for text, LLaVA 1.6 for vision
- **Memory**: Qdrant vector database with sentence-transformers
- **Image Gen**: ComfyUI with custom EDISON nodes
- **Intent**: Optional Google Coral TPU for edge acceleration
- **Frontend**: Modern vanilla JavaScript with responsive design

### Use Cases

- **Enterprise AI** - Deploy private AI for sensitive business operations
- **Research** - Run AI experiments without cloud dependencies
- **Development** - Local AI coding assistant and code generation
- **Education** - Teach AI concepts with full system access
- **Content Creation** - Generate text and images entirely offline
- **Home Lab** - Self-hosted AI assistant for personal projects

### Performance

- Fast responses (<1s for 14B model) with local inference
- Deep analysis with 72B model for complex reasoning
- Vision model processes images with detailed descriptions
- Multi-GPU tensor splitting for larger models
- Automatic model selection based on task complexity

---

**License**: MIT  
**Status**: Active Development  
**Platform**: Ubuntu 22.04+, NVIDIA GPU, CUDA 12+
