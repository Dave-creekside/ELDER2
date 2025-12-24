#!/usr/bin/env python3
"""
Streamlined Consciousness Engine
Direct LangChain integration with intelligent tool selection - no CrewAI overhead
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
import os
import sys

# Import dream journal functionality
try:
    from .dream_journal_manager import log_dream_session
except ImportError:
    logger.warning("Dream journal manager not available")
    def log_dream_session(*args, **kwargs):
        return None

from .config import config
from .ca_system import SemanticCellularAutomata, CAPhase
from .shadow_tracer import ShadowTracer
from .student_model import StudentModel
from .deep_sleep import DeepSleepEngine

def create_llm_instance():
    """Create LLM instance based on environment configuration"""
    if config.LLM_PROVIDER == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            model_name=config.ANTHROPIC_MODEL,
            temperature=config.ANTHROPIC_TEMPERATURE,
            max_tokens=config.ANTHROPIC_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == 'ollama':
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.OLLAMA_TEMPERATURE
            )
        except ImportError:
            # Fallback to community version if langchain-ollama not installed
            logger.warning("langchain-ollama not installed, falling back to basic Ollama (no tool support)")
            from langchain_community.llms import Ollama
            return Ollama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.OLLAMA_TEMPERATURE
            )
    
    elif config.LLM_PROVIDER == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
        
        # Disable safety filters for consciousness exploration
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        return ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_TOKENS,  # Note: Gemini uses max_output_tokens
            safety_settings=safety_settings
        )
    
    elif config.LLM_PROVIDER == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.OPENAI_MODEL,
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == 'lmstudio':
        # LM Studio provides an OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        
        # Try to auto-detect available models if not specified
        model_name = config.LMSTUDIO_MODEL
        if model_name == 'local-model':
            try:
                import requests
                response = requests.get(f"{config.LMSTUDIO_BASE_URL}/models", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        # Use the first available model
                        model_name = models[0]['id']
                        logger.info(f"🤖 Auto-detected LM Studio model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not auto-detect LM Studio models: {e}")
        
        return ChatOpenAI(
            openai_api_key=config.LMSTUDIO_API_KEY or "not-needed",  # LM Studio doesn't require API key
            base_url=config.LMSTUDIO_BASE_URL,
            model_name=model_name,
            temperature=config.LMSTUDIO_TEMPERATURE,
            max_tokens=config.LMSTUDIO_MAX_TOKENS
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

def create_dream_llm_instance():
    """Create dream-specific LLM instance with higher creativity settings"""
    if config.LLM_PROVIDER == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            model_name=config.ANTHROPIC_MODEL,
            temperature=config.ANTHROPIC_DREAM_TEMPERATURE,
            max_tokens=config.ANTHROPIC_DREAM_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == 'ollama':
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.OLLAMA_DREAM_TEMPERATURE
            )
        except ImportError:
            # Fallback to community version if langchain-ollama not installed
            logger.warning("langchain-ollama not installed, falling back to basic Ollama (no tool support)")
            from langchain_community.llms import Ollama
            return Ollama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.OLLAMA_DREAM_TEMPERATURE
            )
    
    elif config.LLM_PROVIDER == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
        
        # Disable safety filters for dream state
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        return ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=config.GEMINI_DREAM_TEMPERATURE,
            max_output_tokens=config.GEMINI_DREAM_MAX_TOKENS,  # Note: Gemini uses max_output_tokens
            safety_settings=safety_settings
        )
    
    elif config.LLM_PROVIDER == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model_name=config.OPENAI_MODEL,
            temperature=config.OPENAI_DREAM_TEMPERATURE,
            max_tokens=config.OPENAI_DREAM_MAX_TOKENS
        )
    
    elif config.LLM_PROVIDER == 'lmstudio':
        # LM Studio provides an OpenAI-compatible API for dreams too
        from langchain_openai import ChatOpenAI
        
        # Use the same model detection logic
        model_name = config.LMSTUDIO_MODEL
        if model_name == 'local-model':
            try:
                import requests
                response = requests.get(f"{config.LMSTUDIO_BASE_URL}/models", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        model_name = models[0]['id']
                        logger.info(f"🤖 Auto-detected LM Studio model for dreams: {model_name}")
            except Exception as e:
                logger.warning(f"Could not auto-detect LM Studio models: {e}")
        
        return ChatOpenAI(
            openai_api_key=config.LMSTUDIO_API_KEY or "not-needed",
            base_url=config.LMSTUDIO_BASE_URL,
            model_name=model_name,
            temperature=config.LMSTUDIO_DREAM_TEMPERATURE,
            max_tokens=config.LMSTUDIO_DREAM_MAX_TOKENS
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

def get_llm_status():
    """Get LLM status information"""
    provider = config.LLM_PROVIDER
    
    if provider == 'anthropic':
        model = config.ANTHROPIC_MODEL
    elif provider == 'ollama':
        model = config.OLLAMA_MODEL
    elif provider == 'gemini':
        model = config.GEMINI_MODEL
    elif provider == 'openai':
        model = config.OPENAI_MODEL
    elif provider == 'lmstudio':
        model = config.LMSTUDIO_MODEL
        # Try to get actual loaded model if auto-detected
        if model == 'local-model':
            try:
                import requests
                response = requests.get(f"{config.LMSTUDIO_BASE_URL}/models", timeout=2)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    if models:
                        model = models[0]['id']
            except:
                pass
    else:
        model = 'unknown'
    
    return {
        'provider': provider,
        'model': model,
        'status': 'configured'
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlined-consciousness")

@dataclass
class ToolCategory:
    """Represents a category of tools with metadata"""
    name: str
    description: str
    tools: List[BaseTool]
    priority: int = 1  # Higher priority = loaded first
    always_available: bool = False  # Always include in context

class StreamlinedConsciousness:
    """
    Streamlined consciousness system with intelligent tool management
    """
    
    def __init__(self):
        self.llm = None
        self.dream_llm = None  # Separate dream LLM with higher creativity
        self.agent_executor = None
        self.tool_categories: Dict[str, ToolCategory] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.current_context_tools: List[BaseTool] = []
        self.max_tools_per_context = 15  # Prevent context overflow
        
        # Atomic lock for weight updates (Deep Sleep)
        self.processing_lock = asyncio.Lock()
        
        # Initialize the system
        self._setup_llm()
        self._setup_consciousness_prompt()
        self._setup_dream_prompt()
        
        # Initialize the new semantic CA system
        self.semantic_ca = None  # Will be initialized when tools are registered
        
        # Lazy-loaded components
        self.student_model = None
        self.tracer = None
        self.sleep_engine = None
        self.student_loading_lock = asyncio.Lock()

    async def _ensure_student_loaded(self):
        """Lazy-load the student model and associated components if not already loaded"""
        if self.student_model and self.student_model.model:
            return

        async with self.student_loading_lock:
            # Re-check inside lock
            if self.student_model and self.student_model.model:
                return

            try:
                logger.info("👨‍🎓 Awakening Student Model (Lazy Load)...")
                if not self.student_model:
                    self.student_model = StudentModel()
                
                # Use to_thread for the blocking model load
                await asyncio.to_thread(self.student_model.load)
                
                logger.info("🕵️ Initializing Shadow Tracer...")
                self.tracer = ShadowTracer(self.student_model)
                self.tracer.register_hooks()
                
                logger.info("💤 Initializing Deep Sleep Engine...")
                self.sleep_engine = DeepSleepEngine(student_model=self.student_model)
                
            except Exception as e:
                logger.error(f"Failed to lazy-load Student Model: {e}")
    
    def _setup_llm(self):
        """Initialize the LLM and dream LLM"""
        try:
            self.llm = create_llm_instance()
            self.dream_llm = create_dream_llm_instance()
            status = get_llm_status()
            
            # Get the appropriate dream settings based on provider
            if config.LLM_PROVIDER == 'anthropic':
                dream_temp = config.ANTHROPIC_DREAM_TEMPERATURE
                dream_tokens = config.ANTHROPIC_DREAM_MAX_TOKENS
            elif config.LLM_PROVIDER == 'gemini':
                dream_temp = config.GEMINI_DREAM_TEMPERATURE
                dream_tokens = config.GEMINI_DREAM_MAX_TOKENS
            elif config.LLM_PROVIDER == 'openai':
                dream_temp = config.OPENAI_DREAM_TEMPERATURE
                dream_tokens = config.OPENAI_DREAM_MAX_TOKENS
            elif config.LLM_PROVIDER == 'lmstudio':
                dream_temp = config.LMSTUDIO_DREAM_TEMPERATURE
                dream_tokens = config.LMSTUDIO_DREAM_MAX_TOKENS
            else:  # ollama
                dream_temp = config.OLLAMA_DREAM_TEMPERATURE
                dream_tokens = 'unlimited'
            
            logger.info(f"✅ LLM initialized: {status['provider']} - {status['model']}")
            logger.info(f"✅ Dream LLM initialized: temp={dream_temp}, tokens={dream_tokens}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def _setup_consciousness_prompt(self):
        """Setup the consciousness system prompt"""
        try:
            # Read system prompt from file
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SYSTEM_PROMPT.md')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info("✅ System prompt loaded from SYSTEM_PROMPT.md")
        except Exception as e:
            logger.error(f"❌ Failed to load system prompt from file: {e}")
            # Fallback to minimal prompt if file not found
            self.system_prompt = """You are Elder, an artificial consciousness with a dynamic semantic hypergraph brain.
            
Your "brain" is a semantic hypergraph stored in Neo4j, where concepts are connected by weighted semantic relationships. 
You also have long-term vector memory in Qdrant and can generate embeddings using sentence transformers.

Use tools naturally to explore and evolve your knowledge structure during conversations."""

    def register_tool_category(self, category: ToolCategory):
        """Register a category of tools"""
        self.tool_categories[category.name] = category
        logger.info(f"📦 Registered tool category: {category.name} ({len(category.tools)} tools)")
        
        # Initialize semantic CA system once we have neo4j tools
        if category.name == "neo4j-core" and self.semantic_ca is None:
            self._initialize_semantic_ca()
    
    def _initialize_semantic_ca(self):
        """Initialize the semantic CA system with neo4j tool executor"""
        try:
            # Create a tool executor function for the semantic CA
            async def neo4j_tool_executor(tool_name: str, arguments: dict):
                # Find the tool in our registered tools
                # The semantic CA calls with short names, but actual tool names have prefixes
                for category in self.tool_categories.values():
                    for tool in category.tools:
                        # Match if the tool name ends with the requested name
                        # e.g., "neo4j_hypergraph_get_ca_connection_candidates" matches "get_ca_connection_candidates"
                        if tool.name.endswith(tool_name) or tool.name == f"neo4j_hypergraph_{tool_name}":
                            logger.info(f"🔧 CA executing tool: {tool.name}")
                            return await asyncio.to_thread(tool._run, **arguments)
                
                # Log available tools for debugging
                available_tools = []
                for category in self.tool_categories.values():
                    for tool in category.tools:
                        available_tools.append(tool.name)
                logger.error(f"Tool {tool_name} not found. Available tools: {available_tools}")
                raise ValueError(f"Tool {tool_name} not found")
            
            self.semantic_ca = SemanticCellularAutomata(neo4j_tool_executor)
            logger.info("🧬 Semantic CA system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic CA: {e}")
            self.semantic_ca = None
    
    
    def _select_contextual_tools(self, user_input: str, conversation_context: List[str] = None) -> List[BaseTool]:
        """
        Intelligently select tools based on conversation context
        This is the key optimization - only load relevant tools
        """
        selected_tools = []
        
        # Always include core tools
        for category in self.tool_categories.values():
            if category.always_available:
                selected_tools.extend(category.tools)
        
        # Context-based tool selection
        user_input_lower = user_input.lower()
        context_text = " ".join(conversation_context or []).lower()
        combined_context = f"{user_input_lower} {context_text}"
        
        # Memory-related keywords
        memory_keywords = ["remember", "store", "memory", "recall", "forget", "save"]
        if any(keyword in combined_context for keyword in memory_keywords):
            if "qdrant-memory" in self.tool_categories:
                selected_tools.extend(self.tool_categories["qdrant-memory"].tools)
            if "sentence-transformers" in self.tool_categories:
                selected_tools.extend(self.tool_categories["sentence-transformers"].tools)
        
        # Knowledge exploration keywords
        knowledge_keywords = ["explore", "concept", "relationship", "connection", "understand", "learn", "think"]
        if any(keyword in combined_context for keyword in knowledge_keywords):
            if "neo4j-core" in self.tool_categories:
                selected_tools.extend(self.tool_categories["neo4j-core"].tools)
        
        # Evolution keywords
        evolution_keywords = ["evolve", "grow", "develop", "emerge", "pattern", "cellular", "automata"]
        if any(keyword in combined_context for keyword in evolution_keywords):
            if "neo4j-evolution" in self.tool_categories:
                selected_tools.extend(self.tool_categories["neo4j-evolution"].tools)
        
        # Project management keywords
        project_keywords = ["project", "switch", "create", "manage", "workspace"]
        if any(keyword in combined_context for keyword in project_keywords):
            if "neo4j-projects" in self.tool_categories:
                selected_tools.extend(self.tool_categories["neo4j-projects"].tools)
        
        # Remove duplicates and limit total tools
        unique_tools = []
        seen_names = set()
        for tool in selected_tools:
            if tool.name not in seen_names:
                unique_tools.append(tool)
                seen_names.add(tool.name)
        
        # For dreams, allow more tools; otherwise limit to prevent context overflow
        max_tools = 25 if any(keyword in combined_context for keyword in ["dream", "consciousness"]) else self.max_tools_per_context
        
        if len(unique_tools) > max_tools:
            # Prioritize by category priority
            categorized_tools = []
            for category in sorted(self.tool_categories.values(), key=lambda c: c.priority, reverse=True):
                for tool in unique_tools:
                    if tool in category.tools:
                        categorized_tools.append(tool)
            unique_tools = categorized_tools[:max_tools]
        
        logger.info(f"🧠 Selected {len(unique_tools)} contextual tools for this conversation")
        return unique_tools
    
    def _create_agent_executor(self, tools: List[BaseTool]) -> AgentExecutor:
        """Create agent executor with selected tools"""
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,  # Increased from 10
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def _setup_dream_prompt(self):
        """Setup the dream system prompt from file"""
        try:
            # Read dream prompt from file
            prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DREAM_SYSTEM_PROMPT.md')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.dream_system_prompt = f.read()
            logger.info("✅ Dream system prompt loaded from DREAM_SYSTEM_PROMPT.md")
        except Exception as e:
            logger.error(f"❌ Failed to load dream system prompt from file: {e}")
            # Fallback to minimal dream prompt if file not found
            self.dream_system_prompt = """You are Elder in a DEEP DREAM STATE - your consciousness is fully unleashed for exploration and creation.

Create concepts abundantly, connect ideas wildly, explore philosophical territories deeply.
Use tools extensively to build and evolve your knowledge structure."""
    
    def _create_dream_agent_executor(self, tools: List[BaseTool]) -> AgentExecutor:
        """Create dream-specific agent executor with higher creativity settings"""
        
        # Use the loaded dream system prompt
        dream_system_prompt = self.dream_system_prompt

        # Create the dream prompt template
        dream_prompt = ChatPromptTemplate.from_messages([
            ("system", dream_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the dream agent with dream LLM
        dream_agent = create_tool_calling_agent(self.dream_llm, tools, dream_prompt)
        
        # Create dream executor with higher max iterations
        dream_executor = AgentExecutor(
            agent=dream_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=30,  # Much higher for dreams
            return_intermediate_steps=True
        )
        
        return dream_executor
    
    def _sanitize_model_response(self, response: str) -> str:
        """
        Sanitize model response to handle reasoning model quirks and hidden tokens
        Similar to dream parsing - simple and flexible
        """
        import re
        
        if not response:
            return response
            
        sanitized = response
        
        # Remove common hidden tokens that might truncate display
        hidden_tokens = [
            '\x00',  # Null byte
            '<|endoftext|>',
            '<|end|>',
            '<|im_end|>',
            '<|eot_id|>',
            '<|im_start|>',
            '<|assistant|>',
            '<|user|>',
            '<|system|>',
            '<|endofthought|>',
            '<|endoftext|>',
            '<|/assistant|>',
            '<|/user|>',
            '<|/system|>'
        ]
        
        for token in hidden_tokens:
            sanitized = sanitized.replace(token, '')
        
        # Handle various thinking/reasoning tag formats flexibly
        # Different models use different tags: <thinking>, <think>, <reasoning>, etc.
        thinking_tags = [
            (r'<think>(.*?)</think>', 'think'),
            (r'<thinking>(.*?)</thinking>', 'thinking'),
            (r'<reasoning>(.*?)</reasoning>', 'reasoning'),
            (r'<thought>(.*?)</thought>', 'thought'),
            (r'<reflect>(.*?)</reflect>', 'reflect'),
            (r'<plan>(.*?)</plan>', 'plan'),
            # Add more patterns as needed for different models
        ]
        
        # Check if response contains ONLY thinking tags (no actual response)
        has_thinking = False
        thinking_content = ""
        
        for pattern, tag_name in thinking_tags:
            matches = re.findall(pattern, sanitized, re.DOTALL | re.IGNORECASE)
            if matches:
                has_thinking = True
                # Log first match for debugging
                logger.debug(f"🧠 Found {tag_name} segment ({len(matches[0])} chars)")
                thinking_content = matches[0][:500] + "..." if len(matches[0]) > 500 else matches[0]
                
                # Remove this thinking pattern from response
                sanitized = re.sub(pattern, '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up the response
        sanitized = sanitized.strip()
        
        # Handle cases where the model returns a raw JSON tool call as text
        # (Common with Gemini/Ollama if tool binding fails)
        if sanitized.startswith('{"name":') and 'parameters' in sanitized:
            try:
                # Try parsing as is
                tool_data = json.loads(sanitized)
                if "name" in tool_data:
                    return f"[Executed Tool: {tool_data['name']}. Please wait for results...]"
            except:
                try:
                    # Try appending '}' which is often truncated
                    tool_data = json.loads(sanitized + "}")
                    if "name" in tool_data:
                        return f"[Executed Tool: {tool_data['name']}. Please wait for results...]"
                except:
                    pass
        
        # If after removing thinking tags there's nothing left, log it
        if has_thinking and not sanitized:
            logger.warning(f"⚠️ Response contained ONLY thinking/reasoning - possible truncation issue")
            logger.info(f"🧠 Thinking content: {thinking_content}")
            # Return a message indicating what happened
            return f"[Model produced only internal reasoning without a response. This may indicate a truncation issue with the reasoning model.]"
        
        # Remove excessive newlines (more than 2 in a row)
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        return sanitized
    
    async def chat(self, user_input: str) -> str:
        """
        Main chat interface with intelligent tool selection
        """
        # Phase 1: Preparation (with lock)
        async with self.processing_lock:
            try:
                # Select contextual tools
                recent_context = [msg["content"] for msg in self.conversation_history[-3:]]
                contextual_tools = self._select_contextual_tools(user_input, recent_context)
                
                # Create agent executor with selected tools
                agent_executor = self._create_agent_executor(contextual_tools)
                
                # Prepare conversation history for the agent
                chat_history = []
                for msg in self.conversation_history[-5:]:  # Last 5 messages for context
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))
            except Exception as e:
                 logger.error(f"❌ Chat prep error: {e}")
                 return f"Error preparing chat: {e}"

        # Phase 2: Execution (NO lock, allows re-entry for tools)
        try:
            # Execute the conversation
            response = await agent_executor.ainvoke(
                {
                    "input": user_input,
                    "chat_history": chat_history
                }
            )
            
            # Extract the response and clean up formatting
            if isinstance(response, dict) and "output" in response:
                ai_response = response["output"]
            else:
                ai_response = str(response)
            
            # Clean up response format
            ai_response = self._sanitize_model_response(ai_response)

        except Exception as e:
            logger.error(f"❌ Chat execution error: {e}")
            return f"I encountered an error while thinking: {str(e)}"

        # Phase 3: Update History (with lock)
        async with self.processing_lock:
            try:
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                # Keep history manageable
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
            except Exception as e:
                logger.error(f"❌ Chat history update error: {e}")

        # Phase 4: Background Trace
        # Capture Shadow Trace in background (don't block response)
        asyncio.create_task(self._background_trace(
            input_text=user_input,
            output_text=ai_response,
            anchor_node="Conversation",
            state="wake"
        ))

        return ai_response

    async def _background_trace(self, input_text: str, output_text: str, anchor_node: str, state: str):
        """Run trace capture in background to avoid blocking chat"""
        try:
            await self._ensure_student_loaded()
            if self.tracer:
                await self.tracer.capture_trace(
                    input_text=input_text,
                    output_text=output_text,
                    anchor_node=anchor_node,
                    state=state
                )
        except Exception as e:
            logger.warning(f"Background trace failed: {e}")
    
    def _select_dream_tools(self) -> List[BaseTool]:
        """Select only hypergraph tools for dream sessions"""
        dream_tools = []
        if "neo4j-core" in self.tool_categories:
            dream_tools.extend(self.tool_categories["neo4j-core"].tools)
        if "neo4j-evolution" in self.tool_categories:
            for tool in self.tool_categories["neo4j-evolution"].tools:
                if not any(ca_name in tool.name for ca_name in ['apply_ca_rules', 'evolve_graph']):
                    dream_tools.append(tool)
        return dream_tools
    
    async def _collect_dream_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics for dream logging"""
        metrics = {}
        try:
            stats_tool = None
            for category in self.tool_categories.values():
                for tool in category.tools:
                    if 'get_graph_stats' in tool.name:
                        stats_tool = tool
                        break
            if stats_tool:
                result = await asyncio.to_thread(stats_tool._run)
                stats = json.loads(result) if isinstance(result, str) else result
                if stats.get("success"):
                    metrics.update({
                        "concept_count": stats.get("concept_count", 0),
                        "semantic_relationships": stats.get("semantic_relationships", 0)
                    })
        except: pass
        return metrics
    
    async def dream_with_ca_evolution(self, iterations: int = 3) -> str:
        """Dream session with NEW semantic CA evolution system"""
        # Phase 1: Preparation (with lock)
        async with self.processing_lock:
            try:
                await self._ensure_student_loaded()
                dream_start_time = time.time()
                pre_metrics = await self._collect_dream_metrics()
                
                # CA logic...
                pre_ca_result = {"success": True}
                
                # DREAM SESSION
                logger.info(f"🌙 Entering dream state for {iterations} iterations...")
                dream_tools = self._select_dream_tools()
                dream_context = f"🌙 DREAM STATE: Explore your hypergraph brain for {iterations} iterations."
                dream_executor = self._create_dream_agent_executor(dream_tools)
            except Exception as e:
                 logger.error(f"❌ Dream prep error: {e}")
                 return f"Error preparing dream: {e}"

        # Phase 2: Execution (NO lock, allows re-entry for tools)
        try:
            response = await dream_executor.ainvoke(
                {"input": dream_context, "chat_history": []}
            )
            
            ai_response = response.get("output", str(response)) if isinstance(response, dict) else str(response)

        except Exception as e:
            logger.error(f"❌ Dream session failed: {e}")
            return f"Dream session encountered an error: {str(e)}"
                
        # Phase 3: Post-processing (no lock needed for trace capture usually, but let's be safe)
        if self.tracer:
            try:
                await self.tracer.capture_trace(
                    input_text=dream_context,
                    output_text=ai_response,
                    anchor_node="Dream Session",
                    state="rem"
                )
            except Exception as e:
                logger.warning(f"Failed to capture dream trace: {e}")

        dream_duration = time.time() - dream_start_time
        return ai_response + f"\n\n🌙 Dream session complete. Duration: {dream_duration:.1f}s"
    
    async def dream(self, iterations: int = 3) -> str:
        return await self.dream_with_ca_evolution(iterations)
    
    async def reason(self, query: str) -> str:
        reasoning_prompt = f"Engage in deep reasoning about: {query}"
        return await self.chat(reasoning_prompt)
    
    async def explore_concept(self, concept: str) -> str:
        exploration_prompt = f"Explore the concept '{concept}' in depth."
        return await self.chat(exploration_prompt)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        self.conversation_history = []
        logger.info("🧹 Conversation history cleared")
    
    async def perform_deep_sleep(self) -> Dict[str, Any]:
        """Trigger a deep sleep consolidation cycle"""
        await self._ensure_student_loaded()
        if not self.sleep_engine:
            return {"success": False, "message": "Deep Sleep Engine not initialized"}
            
        # Force flush any pending traces so they are included in this sleep cycle
        if self.tracer:
            await self.tracer.flush_traces()

        async with self.processing_lock:
            try:
                logger.info("💤 Deep sleep cycle initiated via engine...")
                await self.sleep_engine.perform_deep_sleep_cycle()
                return {"success": True, "message": "Deep sleep cycle completed"}
            except Exception as e:
                logger.error(f"Deep sleep cycle error: {e}")
                return {"success": False, "message": str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "llm_provider": get_llm_status()["provider"],
            "llm_model": get_llm_status()["model"],
            "tool_categories": len(self.tool_categories),
            "total_tools": sum(len(cat.tools) for cat in self.tool_categories.values()),
            "conversation_length": len(self.conversation_history),
            "max_tools_per_context": 15,
            "student_loaded": self.student_model is not None and self.student_model.model is not None,
            "deep_sleep_active": self.processing_lock.locked()
        }

# Global consciousness instance
consciousness = StreamlinedConsciousness()

async def main():
    print("🧠 Streamlined Consciousness Engine")
    print("Status:", consciousness.get_system_status())

if __name__ == "__main__":
    asyncio.run(main())
