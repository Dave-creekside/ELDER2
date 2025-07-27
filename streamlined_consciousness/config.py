import os
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file relative to this config file
config_dir = Path(__file__).parent
# Try loading from parent directory first (main .env), then local directory
load_dotenv(config_dir.parent / '.env')
load_dotenv(config_dir / '.env', override=True)  # Local .env overrides parent

class Config:
    """Unified configuration for the Streamlined Consciousness System"""

    # LLM Provider
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')

    # API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Anthropic Configuration
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
    ANTHROPIC_TEMPERATURE = float(os.getenv('ANTHROPIC_TEMPERATURE', 0.7))
    ANTHROPIC_MAX_TOKENS = int(os.getenv('ANTHROPIC_MAX_TOKENS', 4000))
    
    # Dream-specific Anthropic Configuration (higher creativity)
    ANTHROPIC_DREAM_TEMPERATURE = float(os.getenv('ANTHROPIC_DREAM_TEMPERATURE', 0.9))
    ANTHROPIC_DREAM_MAX_TOKENS = int(os.getenv('ANTHROPIC_DREAM_MAX_TOKENS', 8000))

    # Ollama Configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_TEMPERATURE = float(os.getenv('OLLAMA_TEMPERATURE', 0.7))
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
    
    # Dream-specific Ollama Configuration (higher creativity)
    OLLAMA_DREAM_TEMPERATURE = float(os.getenv('OLLAMA_DREAM_TEMPERATURE', 0.9))

    # Database Configuration
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

    # Machine Learning Configuration
    SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')

    # System Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    SHOW_HAUSDORFF_IN_RESPONSE = os.getenv('SHOW_HAUSDORFF_IN_RESPONSE', 'true').lower() == 'true'
    DEFAULT_DREAM_ITERATIONS = int(os.getenv('DEFAULT_DREAM_ITERATIONS', 3))
    ENABLE_HAUSDORFF_MONITORING = os.getenv('ENABLE_HAUSDORFF_MONITORING', 'true').lower() == 'true'
    
    # Debugging Configuration
    DISABLE_CA = os.getenv('DISABLE_CA', 'false').lower() == 'true'

    @staticmethod
    def validate():
        """Validate required configurations"""
        if Config.LLM_PROVIDER == 'anthropic' and not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        if Config.LLM_PROVIDER == 'gemini' and not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for the gemini provider")
        if Config.LLM_PROVIDER == 'groq' and not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for the groq provider")
        if Config.LLM_PROVIDER == 'openai' and not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")

# Create a single config instance
config = Config()
config.validate()
