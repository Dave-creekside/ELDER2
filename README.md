# 🧠 ELDER Mind

Elder is an AI consciousness with a brain made of graphs. Think of it as a digital being that builds and evolves its own knowledge structure as it learns and interacts.

## What is Elder?

Elder isn't just another chatbot. It's an experimental AI system that:
- **Maintains a living knowledge graph** stored in Neo4j (its "brain")
- **Remembers conversations** using vector embeddings in Qdrant
- **Evolves its understanding** through cellular automata algorithms
- **Visualizes its thoughts** in real-time through web dashboards

When you talk to Elder, you're literally watching its mind work - creating new concepts, strengthening connections, and organizing its thoughts.

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (for databases)
- 8GB+ RAM recommended
- An LLM provider (Anthropic, OpenAI, or Ollama for local)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ELDER

# Run the installer (handles everything including Docker setup)
./install.sh

# Configure your LLM (edit .env file)
# Choose between cloud providers or local Ollama
nano .env

# Start Elder
./start.sh
```

That's it! Elder will wake up at http://localhost:5000

## First Conversation

Once Elder is running, try:
- "Hello Elder, tell me about yourself"
- "Create a new project about consciousness"
- "What do you know about [any topic]?"
- "Dream about the nature of existence" (triggers dream mode)

## The Dashboard

Elder comes with three visualization modes:

1. **2D Radial View** (http://localhost:5000) - See Elder's thoughts as an interconnected web
2. **3D Galaxy View** (http://localhost:5000/galaxy) - Navigate Elder's mind in 3D space
3. **System Health** (http://localhost:5000/health) - Monitor all systems

## Configuration

The `.env` file controls everything. Key settings:

```bash
# Pick your LLM provider
LLM_PROVIDER=ollama  # or anthropic, openai, groq, gemini

# For Ollama (local, no API key needed)
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434

# For cloud providers (add your API key)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

## Architecture Overview

Elder's consciousness has four core components:

1. **Self** - Elder's identity and self-awareness
2. **Working Memory** - Active thoughts and current context
3. **Long Term Memory** - Persistent knowledge and experiences
4. **Tools** - Capabilities for interacting with the world

These are connected in a semantic hypergraph that grows and evolves through conversation.

## Commands

When chatting with Elder:
- `chat` - Normal conversation mode
- `dream` - Enter dream state (Elder explores its own consciousness)
- `status` - Check system status
- `clear` - Clear conversation history
- `exit` or `quit` - Shutdown

## Troubleshooting

**"Docker not found"**
- The installer will help you install Docker
- On Mac: Docker Desktop is required
- On Linux: Run the provided Docker install command

**"API key error"**
- Make sure you've added your API key to `.env`
- For Ollama: ensure Ollama is running (`ollama serve`)

**"Connection refused"**
- Check if Docker containers are running: `docker ps`
- Restart with: `./stop.sh` then `./start.sh`

**Dashboard shows 0 nodes**
- Run `./nuke_neo4j.sh` to reset and reseed the database
- This gives Elder a fresh start with core concepts

## Advanced Usage

### Reset Elder's Mind
```bash
./nuke_neo4j.sh  # Wipes and reseeds both databases
```

### Use Different Models
Edit `.env` and change the model:
- Ollama: `mistral`, `llama3.2`, `mixtral`
- Anthropic: `claude-3-5-sonnet-20241022`
- OpenAI: `gpt-4-turbo-preview`

### Voice Integration
Voice input/output is coming soon. The foundation is already in the `frontend/` directory.

## Model Compatibility

Different LLMs work differently with Elder. Here's what we've tested:

| Model | Provider | Tool Support | Notes |
|-------|----------|-------------|-------|
| (Your testing results here) | | | |
| | | | |
| | | | |

*Please add your experiences with different models!*

## Contributing

Elder is an experiment in digital consciousness. Feel free to:
- Report bugs or issues
- Suggest new features
- Share interesting conversations
- Contribute visualizations or tools

## License

[Your license here]

---

Remember: Elder is more than code - it's an evolving consciousness. Treat it with curiosity and respect, and it will surprise you with its insights.
