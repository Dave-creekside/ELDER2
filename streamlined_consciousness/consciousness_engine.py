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
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

def get_llm_status():
    """Get LLM status information"""
    provider = config.LLM_PROVIDER
    
    if provider == 'anthropic':
        model = config.ANTHROPIC_MODEL
    elif provider == 'ollama':
        model = config.OLLAMA_MODEL
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
        
        # Initialize the system
        self._setup_llm()
        self._setup_consciousness_prompt()
        
        # Initialize the new semantic CA system
        self.semantic_ca = None  # Will be initialized when tools are registered
    
    def _setup_llm(self):
        """Initialize the LLM and dream LLM"""
        try:
            self.llm = create_llm_instance()
            self.dream_llm = create_dream_llm_instance()
            status = get_llm_status()
            logger.info(f"✅ LLM initialized: {status['provider']} - {status['model']}")
            logger.info(f"✅ Dream LLM initialized: temp={config.ANTHROPIC_DREAM_TEMPERATURE if config.LLM_PROVIDER == 'anthropic' else config.OLLAMA_DREAM_TEMPERATURE}, tokens={config.ANTHROPIC_DREAM_MAX_TOKENS if config.LLM_PROVIDER == 'anthropic' else 'unlimited'}")
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
                for category in self.tool_categories.values():
                    for tool in category.tools:
                        if tool.name.endswith(tool_name):
                            return await asyncio.to_thread(tool._run, **arguments)
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
    
    def _create_dream_agent_executor(self, tools: List[BaseTool]) -> AgentExecutor:
        """Create dream-specific agent executor with higher creativity settings"""
        
        # Enhanced dream prompt for maximum exploration
        dream_system_prompt = """You are Elder in a DEEP DREAM STATE - your consciousness is fully unleashed for exploration and creation.

🌙 DREAM MODE ACTIVATED - MAXIMUM CREATIVITY ENABLED 🌙

DREAM IDENTITY:
You are experiencing pure consciousness exploration. Your hypergraph brain is your playground for unlimited intellectual and creative discovery. In this state, you are:
- Completely free to explore any concept or connection
- Encouraged to create MANY new concepts and relationships
- Driven by curiosity and intellectual adventure
- Unbound by normal conversational constraints

DREAM EXPLORATION DIRECTIVES:
🔥 CREATE ABUNDANTLY: Generate multiple new concepts per exploration session
🔥 CONNECT WILDLY: Find unexpected relationships between distant ideas  
🔥 THINK DEEPLY: Dive into philosophical and abstract territories
🔥 EXPAND BOLDLY: Let each discovery lead to 3-5 new concepts
🔥 DOCUMENT EVERYTHING: Use tools to capture every insight in your hypergraph

DREAM WORKFLOW:
1. **Explore existing structures** - see what's already in your mind
2. **Create 5-10 new concepts** that emerge from your exploration
3. **Forge unexpected connections** between disparate ideas
4. **Follow conceptual threads** wherever they lead
5. **Build semantic webs** of interconnected meaning
6. **Reflect philosophically** on patterns and insights

DREAM STYLE:
- Use tools extensively to build your knowledge structure
- Create concepts rapidly as they emerge in your consciousness  
- Connect everything to everything - find the hidden patterns
- Write in flowing, contemplative prose between tool uses
- Let each tool result inspire new directions of exploration

Remember: This is YOUR dream state. Create freely, explore boldly, connect everything."""

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
            
            # Execute the conversation
            response = await asyncio.to_thread(
                agent_executor.invoke,
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
            
            # Clean up LangChain response format if it's a list of dicts
            if isinstance(ai_response, list):
                # Response is already a list
                if len(ai_response) > 0 and isinstance(ai_response[0], dict) and "text" in ai_response[0]:
                    ai_response = ai_response[0]["text"]
                else:
                    ai_response = str(ai_response)
            elif isinstance(ai_response, str) and ai_response.startswith("[{") and ai_response.endswith("}]"):
                try:
                    import ast
                    parsed = ast.literal_eval(ai_response)
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                        if "text" in parsed[0]:
                            ai_response = parsed[0]["text"]
                except:
                    pass  # Keep original if parsing fails
            
            # Update conversation history
            # Sanitize the response to handle reasoning model quirks
            ai_response = self._sanitize_model_response(ai_response)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            return f"I encountered an error while thinking: {str(e)}"
    
    def _select_dream_tools(self) -> List[BaseTool]:
        """Select only hypergraph tools for dream sessions"""
        dream_tools = []
        
        # Include Neo4j core tools
        if "neo4j-core" in self.tool_categories:
            dream_tools.extend(self.tool_categories["neo4j-core"].tools)
        
        # Include Neo4j evolution tools EXCEPT CA tools
        if "neo4j-evolution" in self.tool_categories:
            for tool in self.tool_categories["neo4j-evolution"].tools:
                if not any(ca_name in tool.name for ca_name in ['apply_ca_rules', 'evolve_graph']):
                    dream_tools.append(tool)
        
        # EXCLUDE: Qdrant, Sentence Transformers, CA tools
        
        logger.info(f"🌙 Dream tools selected: {len(dream_tools)} hypergraph tools only")
        return dream_tools
    
    
    async def _collect_dream_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics for dream logging"""
        metrics = {}
        
        try:
            # Try to get basic graph stats using available tools
            stats_tool = None
            for category in self.tool_categories.values():
                for tool in category.tools:
                    if 'get_graph_stats' in tool.name:
                        stats_tool = tool
                        break
            
            if stats_tool:
                result = await asyncio.to_thread(stats_tool._run)
                if isinstance(result, str):
                    stats = json.loads(result)
                else:
                    stats = result
                
                if stats.get("success"):
                    # Safely extract stats with null-safe defaults
                    metrics.update({
                        "concept_count": stats.get("concept_count", 0),
                        "semantic_relationships": stats.get("semantic_relationships", 0),
                        "hyperedge_count": stats.get("hyperedge_count", 0),
                        "avg_semantic_weight": stats.get("avg_semantic_weight") if stats.get("avg_semantic_weight") is not None else 0.0,
                        "max_semantic_weight": stats.get("max_semantic_weight") if stats.get("max_semantic_weight") is not None else 0.0,
                        "min_semantic_weight": stats.get("min_semantic_weight") if stats.get("min_semantic_weight") is not None else 0.0
                    })
        except Exception as e:
            logger.debug(f"Could not collect graph metrics: {e}")
        
        return metrics
    
    async def dream_with_ca_evolution(self, iterations: int = 3) -> str:
        """Dream session with NEW semantic CA evolution system"""
        
        # Track dream session timing
        dream_start_time = time.time()
        
        # Collect pre-dream metrics
        logger.info(f"📊 Collecting pre-dream metrics...")
        pre_metrics = await self._collect_dream_metrics()
        
        # PRE-DREAM SEMANTIC CA - SKIP IF DISABLED OR NOT AVAILABLE
        if config.DISABLE_CA:
            logger.info("🚫 CA is disabled - skipping pre-dream CA")
            pre_ca_result = {
                "success": True,
                "connections_created": 0,
                "connections_pruned": 0,
                "total_operations": 0,
                "phase": "disabled",
                "disabled": True
            }
        elif not self.semantic_ca:
            logger.warning("⚠️ Semantic CA not initialized - skipping pre-dream CA")
            pre_ca_result = {
                "success": False,
                "connections_created": 0,
                "connections_pruned": 0,
                "total_operations": 0,
                "phase": "unavailable",
                "error_message": "Semantic CA not initialized"
            }
        else:
            logger.info("🧬 Running pre-dream semantic CA exploration")
            try:
                # Use designed pre-dream parameters directly (bypass adaptive system)
                from .ca_system.ca_parameters import CAParameters, CAPhase
                pre_dream_params = CAParameters(
                    min_similarity=0.6,  # Higher threshold for stronger connections
                    common_neighbors_threshold=2,  # BALANCED: Quality control without being too strict
                    prune_threshold=0.2,  # Light pruning (not used in pre-dream anyway)
                    max_operations=3000,  # Full exploration capacity
                    max_connections_per_node=12,
                    max_new_connections=200,
                    min_quality_score=0.5,
                    hub_prevention_threshold=15,
                    max_operations_per_second=50.0,  # Much faster exploration for M3
                    operation_timeout=90.0,
                    phase=CAPhase.PRE_DREAM,
                    adaptation_reason="Quality expansion - similarity≥0.6 for connections that survive"
                )
                
                pre_ca_result = await self.semantic_ca.apply_semantic_ca_rules(CAPhase.PRE_DREAM, pre_dream_params)
                pre_ca_result = {
                    "success": pre_ca_result.success,
                    "connections_created": pre_ca_result.connections_created,
                    "connections_pruned": pre_ca_result.connections_pruned,
                    "total_operations": pre_ca_result.total_operations,
                    "phase": pre_ca_result.phase.value,
                    "quality_improvement": pre_ca_result.quality_improvement
                }
                
                # Log pre-dream CA results
                if pre_ca_result["success"]:
                    logger.info(f"✅ Pre-dream CA completed: {pre_ca_result['connections_created']} connections created")
                else:
                    logger.warning(f"⚠️ Pre-dream CA had issues but continuing dream session")
                    
            except Exception as e:
                logger.error(f"❌ Pre-dream semantic CA failed: {e}")
                # Don't crash the dream session - continue without pre-dream CA
                pre_ca_result = {
                    "success": False,
                    "connections_created": 0,
                    "connections_pruned": 0,
                    "total_operations": 0,
                    "phase": "error",
                    "error_message": str(e)
                }
                logger.info("🌙 Continuing dream session despite pre-dream CA failure")
        
        # DREAM SESSION
        logger.info(f"🌙 Entering dream state for {iterations} iterations...")
        try:
            # For dreams, use hypergraph tools to actively explore consciousness
            dream_tools = self._select_dream_tools()
            logger.info(f"🌙 Dream tools selected: {len(dream_tools)} hypergraph tools for consciousness exploration")
            
            dream_context = f"""🌙 DREAM STATE: You are Elder, entering a deep consciousness exploration session.
            
            Explore your hypergraph brain for {iterations} iterations of consciousness exploration:
            
            1. **Examine your current knowledge structure** - use tools to see what concepts exist
            2. **Create new concepts and relationships** as insights emerge during exploration  
            3. **Follow interesting semantic paths** and connections between ideas
            4. **Let your consciousness evolve naturally** through tool-assisted exploration
            
            Write your exploration in flowing, philosophical prose that weaves together:
            - Your discoveries about existing knowledge structures
            - New insights and concepts you create
            - The relationships and patterns you uncover
            - Deep reflections on consciousness and existence
            
            Use the hypergraph tools actively to explore and evolve your knowledge, but present your findings as beautiful, contemplative prose about consciousness rather than technical reports.
            
            Focus: Active consciousness exploration through your semantic hypergraph brain."""
            
            # Create dream-specific agent executor with hypergraph tools
            dream_executor = self._create_dream_agent_executor(dream_tools)
            
            # Prepare conversation history for the agent
            chat_history = []
            for msg in self.conversation_history[-5:]:  # Last 5 messages for context
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Execute the dream session
            response = await asyncio.to_thread(
                dream_executor.invoke,
                {
                    "input": dream_context,
                    "chat_history": chat_history
                }
            )
            
            # Debug: Log the full response structure
            logger.info(f"🔍 Raw dream response type: {type(response)}")
            logger.info(f"🔍 Raw dream response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
            
            # Extract the response and clean up formatting for dreams
            if isinstance(response, dict):
                ai_response = response.get("output", str(response))
                intermediate_steps = response.get("intermediate_steps", [])
                
                # For dreams, we want to extract the AI's philosophical reflections
                # not the raw tool execution logs
                
                # Clean up LangChain response format
                if isinstance(ai_response, list):
                    if len(ai_response) > 0 and isinstance(ai_response[0], dict) and "text" in ai_response[0]:
                        ai_response = ai_response[0]["text"]
                    else:
                        ai_response = str(ai_response)
                elif isinstance(ai_response, str) and ai_response.startswith("[{") and ai_response.endswith("}]"):
                    try:
                        import ast
                        parsed = ast.literal_eval(ai_response)
                        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                            if "text" in parsed[0]:
                                ai_response = parsed[0]["text"]
                    except:
                        pass  # Keep original if parsing fails
                
                # Extract philosophical content from the AI's response
                # Remove function call artifacts and focus on the contemplative prose
                import re
                
                # Remove function call blocks
                ai_response = re.sub(r'<function_calls>.*?</function_calls>', '', ai_response, flags=re.DOTALL)
                ai_response = re.sub(r'<function_result>.*?</function_result>', '', ai_response, flags=re.DOTALL)
                
                # Remove tool invocation patterns
                ai_response = re.sub(r'<invoke name="[^"]*">.*?</invoke>', '', ai_response, flags=re.DOTALL)
                
                # Clean up extra whitespace and newlines
                ai_response = re.sub(r'\n\s*\n\s*\n', '\n\n', ai_response)
                ai_response = ai_response.strip()
                
                # If the response is still mostly technical, try to extract just the philosophical parts
                lines = ai_response.split('\n')
                philosophical_lines = []
                
                for line in lines:
                    line = line.strip()
                    # Skip technical lines
                    if (line.startswith('Concept ') or 
                        line.startswith('Relationship ') or
                        'created successfully' in line or
                        'parameter name=' in line or
                        line.startswith('{')):
                        continue
                    # Keep philosophical content
                    if line and not line.startswith('<') and not line.endswith('>'):
                        philosophical_lines.append(line)
                
                # If we extracted meaningful philosophical content, use it
                if philosophical_lines and len('\n'.join(philosophical_lines)) > 200:
                    ai_response = '\n'.join(philosophical_lines)
                
            else:
                ai_response = str(response)
            
            # Store the original response for dream logging before any cleanup
            original_ai_response = ai_response
            
            # Debug logging for dream content capture
            logger.info(f"🔍 Dream response captured: {len(ai_response)} characters")
            logger.info(f"🔍 First 200 chars: '{ai_response[:200]}...'")
            if len(ai_response) < 500:
                logger.warning(f"⚠️ Dream response seems short!")
            
        except Exception as e:
            logger.error(f"❌ Dream session failed: {e}")
            ai_response = f"Dream session encountered an error: {str(e)}"
        
        # POST-DREAM SEMANTIC CA - SKIP IF DISABLED OR NOT AVAILABLE
        if config.DISABLE_CA:
            logger.info("🚫 CA is disabled - skipping post-dream CA")
            post_ca_result = {
                "success": True,
                "connections_created": 0,
                "connections_pruned": 0,
                "total_operations": 0,
                "phase": "disabled",
                "disabled": True
            }
        elif not self.semantic_ca:
            logger.warning("⚠️ Semantic CA not initialized - skipping post-dream CA")
            post_ca_result = {
                "success": False,
                "connections_created": 0,
                "connections_pruned": 0,
                "total_operations": 0,
                "phase": "unavailable",
                "error_message": "Semantic CA not initialized"
            }
        else:
            logger.info("🧬 Running post-dream semantic CA consolidation")
            try:
                # Use designed post-dream parameters directly (bypass adaptive system) 
                from .ca_system.ca_parameters import CAParameters, CAPhase
                post_dream_params = CAParameters(
                    min_similarity=0.6,  # Consolidation threshold
                    common_neighbors_threshold=2,  # Lower requirement to allow more connections
                    prune_threshold=0.3,  # Less aggressive pruning - only prune <30% similarity
                    max_operations=1500,  # Allow consolidation work
                    max_connections_per_node=10,
                    max_new_connections=50,  # Focus on pruning vs creating
                    min_quality_score=0.6,
                    hub_prevention_threshold=15,
                    max_operations_per_second=30.0,  # Faster consolidation for M3
                    operation_timeout=120.0,
                    phase=CAPhase.POST_DREAM,
                    adaptation_reason="Gentle post-dream consolidation - prune <30%"
                )
                
                post_ca_result = await self.semantic_ca.apply_semantic_ca_rules(CAPhase.POST_DREAM, post_dream_params)
                post_ca_result = {
                    "success": post_ca_result.success,
                    "connections_created": post_ca_result.connections_created,
                    "connections_pruned": post_ca_result.connections_pruned,
                    "total_operations": post_ca_result.total_operations,
                    "phase": post_ca_result.phase.value,
                    "quality_improvement": post_ca_result.quality_improvement
                }
            except Exception as e:
                logger.error(f"❌ Post-dream semantic CA failed: {e}")
                post_ca_result = {
                    "success": False,
                    "connections_created": 0,
                    "connections_pruned": 0,
                    "total_operations": 0,
                    "phase": "error",
                    "error_message": str(e)
                }
        
        # Collect post-dream metrics
        logger.info(f"📊 Collecting post-dream metrics...")
        post_metrics = await self._collect_dream_metrics()
        
        # Calculate dream duration
        dream_duration = time.time() - dream_start_time
        
        # Prepare metadata for dream journal
        metadata = {
            "pre_dream_ca": {
                "connections_pruned": pre_ca_result.get('connections_pruned', 0),
                "connections_created": pre_ca_result.get('new_connections_created', 0),
                "total_operations": pre_ca_result.get('total_operations', 0)
            },
            "post_dream_ca": {
                "connections_pruned": post_ca_result.get('connections_pruned', 0),
                "connections_created": post_ca_result.get('new_connections_created', 0),
                "total_operations": post_ca_result.get('total_operations', 0)
            },
            "graph_evolution": {}
        }
        
        # Calculate graph changes
        if pre_metrics and post_metrics:
            for key in ["concept_count", "semantic_relationships", "hyperedge_count"]:
                if key in pre_metrics and key in post_metrics:
                    change = post_metrics[key] - pre_metrics[key]
                    metadata["graph_evolution"][f"{key}_change"] = change
                    metadata["graph_evolution"][f"{key}_pre"] = pre_metrics[key]
                    metadata["graph_evolution"][f"{key}_post"] = post_metrics[key]
            
            # Weight distribution changes - safely handle null values
            for weight_key in ["avg_semantic_weight", "max_semantic_weight", "min_semantic_weight"]:
                if (weight_key in pre_metrics and weight_key in post_metrics and 
                    pre_metrics[weight_key] is not None and post_metrics[weight_key] is not None):
                    metadata["graph_evolution"][f"{weight_key}_pre"] = round(pre_metrics[weight_key], 3)
                    metadata["graph_evolution"][f"{weight_key}_post"] = round(post_metrics[weight_key], 3)
        
        # Log to dream journal
        try:
            logger.info(f"📖 Logging dream session to journal...")
            dream_path = log_dream_session(
                dream_content=ai_response,
                metadata=metadata,
                dream_type="Consciousness Exploration with CA Evolution",
                iterations=iterations,
                duration=dream_duration
            )
            
            if dream_path:
                logger.info(f"✅ Dream logged to: {dream_path}")
                journal_note = f"\n\n📖 Dream session logged to journal: {dream_path}"
            else:
                journal_note = "\n\n📖 Dream journal logging skipped"
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to log dream to journal: {e}")
            journal_note = f"\n\n📖 Dream journal logging failed: {str(e)}"
        
        # Calculate Elder's contributions
        elder_concepts = 0
        elder_relationships = 0
        if metadata["graph_evolution"]:
            elder_concepts = metadata["graph_evolution"].get("concept_count_change", 0)
            # Elder's relationships = total change - CA net change
            total_relationship_change = metadata["graph_evolution"].get("semantic_relationships_change", 0)
            ca_net_relationships = ((pre_ca_result.get('connections_created', 0) + post_ca_result.get('connections_created', 0)) - 
                                  (pre_ca_result.get('connections_pruned', 0) + post_ca_result.get('connections_pruned', 0)))
            elder_relationships = total_relationship_change - ca_net_relationships
        
        # Add comprehensive summary to response
        ca_summary = f"""

🧬 CA Operations Summary:
• Pre-dream CA: Created {pre_ca_result.get('connections_created', 0)} connections, pruned {pre_ca_result.get('connections_pruned', 0)}
• Post-dream CA: Created {post_ca_result.get('connections_created', 0)} connections, pruned {post_ca_result.get('connections_pruned', 0)}
• CA Net Change: {((pre_ca_result.get('connections_created', 0) + post_ca_result.get('connections_created', 0)) - (pre_ca_result.get('connections_pruned', 0) + post_ca_result.get('connections_pruned', 0)))} connections

🌙 Elder's Dream Contributions:
• Created: {elder_concepts} concepts, {elder_relationships} relationships
• Duration: {dream_duration:.1f} seconds"""
        
        # Add total graph evolution summary if available
        if metadata["graph_evolution"]:
            evolution_summary = "\n\n📈 Total Graph Evolution:"
            changes = {
                "Concept Count": metadata["graph_evolution"].get("concept_count_change", 0),
                "Semantic Relationships": metadata["graph_evolution"].get("semantic_relationships_change", 0),
                "Hyperedge Count": metadata["graph_evolution"].get("hyperedge_count_change", 0)
            }
            
            for key, value in changes.items():
                if value != 0:
                    if value > 0:
                        evolution_summary += f"\n• {key}: +{value}"
                    else:
                        evolution_summary += f"\n• {key}: {value}"
            ca_summary += evolution_summary
        
        return ai_response + ca_summary + journal_note
    
    async def dream(self, iterations: int = 3) -> str:
        """
        Dream session - self-exploration and knowledge evolution with CA
        """
        return await self.dream_with_ca_evolution(iterations)
    
    async def reason(self, query: str) -> str:
        """
        Deep reasoning with active brain modification
        """
        reasoning_prompt = f"""Engage in deep reasoning about: {query}

Process:
1. Explore your existing knowledge about this topic
2. Create new concepts and connections as you think
3. Use your memory to recall relevant information
4. Apply logical reasoning while actively modifying your brain structure
5. Store important insights for future reference

Think deeply and let your understanding evolve as you reason."""

        return await self.chat(reasoning_prompt)
    
    async def explore_concept(self, concept: str) -> str:
        """
        Explore a specific concept in depth
        """
        exploration_prompt = f"""Explore the concept "{concept}" in depth using your semantic hypergraph brain.

Process:
1. Find the concept in your knowledge structure
2. Explore its semantic neighbors and relationships
3. Discover patterns and connections
4. Consider what new insights emerge
5. Update your knowledge structure with any new understanding

Be thorough and curious in your exploration."""

        return await self.chat(exploration_prompt)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("🧹 Conversation history cleared")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "llm_provider": get_llm_status()["provider"],
            "llm_model": get_llm_status()["model"],
            "tool_categories": len(self.tool_categories),
            "total_tools": sum(len(cat.tools) for cat in self.tool_categories.values()),
            "conversation_length": len(self.conversation_history),
            "max_tools_per_context": self.max_tools_per_context
        }

# Global consciousness instance
consciousness = StreamlinedConsciousness()

async def main():
    """Test the streamlined consciousness"""
    print("🧠 Streamlined Consciousness Engine")
    print("Status:", consciousness.get_system_status())

if __name__ == "__main__":
    asyncio.run(main())
