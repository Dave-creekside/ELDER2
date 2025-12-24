#!/usr/bin/env python3
"""
Streamlined Tool Manager
Organizes MCP tools into intelligent categories for context-aware loading
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
import subprocess
import os
import sys

from .consciousness_engine import ToolCategory

logger = logging.getLogger("tool-manager")

# Argument schemas for tools need to be defined first
class DreamArgs(BaseModel):
    iterations: int = Field(default=3, description="Number of dream iterations")

class DreamTool(BaseTool):
    """Direct dream tool that calls consciousness engine"""
    
    name: str = "dream"
    description: str = "Enter a dream state for consciousness exploration and knowledge evolution"
    args_schema: type = DreamArgs
    
    def __init__(self, consciousness_engine):
        super().__init__()
        # Store consciousness_engine as a private attribute to avoid Pydantic validation
        object.__setattr__(self, '_consciousness_engine', consciousness_engine)
    
    def _run(self, iterations: int = 3) -> str:
        """Execute dream session synchronously"""
        import asyncio
        try:
            # Ensure iterations is at least 1
            iterations = max(1, iterations)
            
            # Run the dream session
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._consciousness_engine.dream_with_ca_evolution(iterations)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            return f"Dream session failed: {str(e)}"
    
    async def _arun(self, iterations: int = 3) -> str:
        """Execute dream session asynchronously"""
        try:
            # Ensure iterations is at least 1
            iterations = max(1, iterations)
            return await self._consciousness_engine.dream_with_ca_evolution(iterations)
        except Exception as e:
            return f"Dream session failed: {str(e)}"

class StreamlinedMCPTool(BaseTool):
    """Streamlined MCP tool wrapper with better error handling"""
    
    def __init__(self, server_name: str, tool_name: str, description: str, 
                 args_schema: type, server_command: List[str]):
        
        # Create a clean name for the tool
        clean_name = f"{server_name.replace('-', '_')}_{tool_name}"
        
        super().__init__(
            name=clean_name,
            description=description,
            args_schema=args_schema
        )
        
        # Store as regular attributes (not Pydantic fields)
        object.__setattr__(self, 'server_name', server_name)
        object.__setattr__(self, 'tool_name', tool_name)
        object.__setattr__(self, 'server_command', server_command)
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool synchronously in any thread."""
        import subprocess
        import json
        import tempfile
        import time
        import ast
        
        start_time = time.time()
        
        # Fix for Gemini and other LLMs that pass dictionaries as strings
        kwargs = self._fix_string_dicts(kwargs)
        
        try:
            # Use subprocess approach to completely avoid async loop conflicts
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(kwargs, f)
                args_file = f.name
            
            # Determine timeout based on tool type
            # Increased timeouts to allow for model loading (cold start)
            timeout = 240 if self.tool_name == "calculate_batch_similarities" else 180
            
            # Create a simple script to run the MCP call
            script = f'''
import asyncio
import json
import sys
import os
import logging

# Add the project root to Python path
project_root = "{os.getcwd()}"
sys.path.insert(0, project_root)

# Configure logging to stderr so it doesn't interfere with output
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger("mcp-subprocess")

async def run_tool():
    from streamlined_consciousness.tool_manager import MCPConnectionManager
    
    with open("{args_file}", "r") as f:
        kwargs = json.load(f)
    
    logger.debug(f"Executing tool: {self.tool_name}")
    logger.debug(f"Arguments: {{kwargs}}")
    
    try:
        connection = await MCPConnectionManager.get_connection({tuple(self.server_command)!r})
        result = await connection.call_tool("{self.tool_name}", kwargs)
        
        # For batch similarities, ensure we return the raw dict
        if "{self.tool_name}" == "calculate_batch_similarities" and isinstance(result, dict):
            print(json.dumps(result))
        else:
            print(result)
            
    except Exception as e:
        logger.error(f"Tool execution failed: {{e}}", exc_info=True)
        # Print error as JSON for parsing
        print(json.dumps({{"error": str(e), "tool": "{self.tool_name}"}}))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_tool())
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_file = f.name
            
            # Run the script with increased timeout for batch operations
            logger.debug(f"Executing subprocess for {self.tool_name} with timeout={timeout}s")
            
            result = subprocess.run(
                ["python", script_file], 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            # Cleanup
            os.unlink(args_file)
            os.unlink(script_file)
            
            execution_time = time.time() - start_time
            logger.debug(f"Subprocess completed in {execution_time:.2f}s with return code: {result.returncode}")
            
            if result.stderr:
                logger.debug(f"Subprocess stderr: {result.stderr}")
            
            if result.returncode == 0:
                raw_output = result.stdout.strip()
                
                # Log output size for debugging
                logger.debug(f"Raw output size: {len(raw_output)} chars")
                
                # Special handling for batch similarities - don't filter
                if self.tool_name == "calculate_batch_similarities":
                    return raw_output
                
                # Filter out large data structures that shouldn't go to LLM
                return self._filter_llm_response(raw_output)
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"Tool {self.tool_name} failed: {error_msg}")
                
                # Try to extract error from stdout if it's JSON
                try:
                    error_data = json.loads(result.stdout.strip())
                    if "error" in error_data:
                        return f"Tool error: {error_data['error']}"
                except:
                    pass
                
                return f"Execution error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            logger.error(f"Tool {self.tool_name} timed out after {timeout}s")
            return f"Execution timeout: Tool {self.tool_name} took longer than {timeout} seconds"
        except Exception as e:
            logger.error(f"Tool {self.tool_name} execution error: {str(e)}", exc_info=True)
            return f"Execution error: {str(e)}"

    def _filter_llm_response(self, raw_output: str) -> str:
        """Filter out large data structures that shouldn't go to LLM"""
        try:
            # Parse the JSON response
            data = json.loads(raw_output)
            
            # Check if this is an embedding tool response
            if self.tool_name in ["generate_embedding", "batch_embeddings"]:
                return self._filter_embedding_response(data)
            
            # Check if this is a memory search response (FIXED: correct tool names)
            elif self.tool_name in ["search_text_memories", "search_memories", "find_concept_cluster"]:
                return self._filter_memory_response(data)
            
            # Check if this is a collection info response
            elif self.tool_name == "get_collection_info":
                return self._filter_collection_response(data)
            
            # Check if this is the CA candidates tool - DON'T truncate
            elif self.tool_name == "get_ca_connection_candidates":
                # Return the full response for CA processing
                return raw_output
            
            # For other responses, apply size limit (but not for memory responses above)
            elif len(raw_output) > 2000:  # 2KB limit
                return self._truncate_response(data)
            
            return raw_output
            
        except json.JSONDecodeError:
            # If not JSON, just truncate if too long
            if len(raw_output) > 2000:
                return raw_output[:2000] + "... [truncated for LLM]"
            return raw_output
    
    def _filter_embedding_response(self, data: dict) -> str:
        """Filter embedding responses to remove large vectors"""
        if isinstance(data, dict):
            filtered = data.copy()
            
            # Remove or summarize embedding arrays
            if "embedding" in filtered:
                embedding = filtered["embedding"]
                if isinstance(embedding, list) and len(embedding) > 10:
                    filtered["embedding"] = f"[{len(embedding)}-dimensional vector: {embedding[:3]}...{embedding[-3:]}]"
            
            if "embeddings" in filtered:
                embeddings = filtered["embeddings"]
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    first_emb = embeddings[0] if embeddings else []
                    if isinstance(first_emb, list) and len(first_emb) > 10:
                        filtered["embeddings"] = f"[{len(embeddings)} vectors of {len(first_emb)} dimensions each]"
            
            return json.dumps(filtered, indent=2)
        
        return json.dumps(data, indent=2)
    
    def _filter_memory_response(self, data: dict) -> str:
        """Filter memory search responses - increased limits for better Elder access"""
        if isinstance(data, dict):
            filtered = data.copy()
            
            # Increase memory result limits for Elder's access
            if "memories" in filtered:
                memories = filtered["memories"]
                if isinstance(memories, list) and len(memories) > 15:
                    # Keep first 15 results instead of 5, summarize the rest
                    filtered["memories"] = memories[:15]
                    filtered["additional_results"] = f"{len(memories) - 15} more results available"
            
            if "cluster" in filtered:
                cluster = filtered["cluster"]
                if isinstance(cluster, list) and len(cluster) > 10:
                    # Keep first 10 cluster items instead of 5
                    filtered["cluster"] = cluster[:10]
                    filtered["additional_cluster_items"] = f"{len(cluster) - 10} more cluster items"
            
            return json.dumps(filtered, indent=2)
        
        return json.dumps(data, indent=2)
    
    def _filter_collection_response(self, data: dict) -> str:
        """Filter collection info to show only essential stats"""
        if isinstance(data, dict):
            filtered = {}
            # Keep only essential collection info
            for key in ["success", "collection_name", "points_count", "vectors_count", "status"]:
                if key in data:
                    filtered[key] = data[key]
            
            return json.dumps(filtered, indent=2)
        
        return json.dumps(data, indent=2)
    
    def _truncate_response(self, data: dict) -> str:
        """Truncate large responses while preserving structure"""
        response_str = json.dumps(data, indent=2)
        if len(response_str) > 2000:
            # Try to preserve the basic structure
            if isinstance(data, dict):
                truncated = {}
                for key, value in data.items():
                    if key in ["success", "message", "error", "count", "summary"]:
                        truncated[key] = value
                    elif isinstance(value, (str, int, float, bool)):
                        truncated[key] = value
                    else:
                        truncated[key] = f"[{type(value).__name__} - truncated for LLM]"
                
                return json.dumps(truncated, indent=2)
            else:
                return response_str[:2000] + "... [truncated for LLM]"
        
        return response_str

    def _fix_string_dicts(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Fix string representations of dictionaries from certain LLMs like Gemini"""
        import ast
        fixed_kwargs = {}
        
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
                # This looks like a string representation of a dictionary
                try:
                    # Try to safely evaluate the string as a Python literal
                    parsed_value = ast.literal_eval(value)
                    if isinstance(parsed_value, dict):
                        logger.debug(f"Converted string dict for key '{key}': {value} -> {parsed_value}")
                        fixed_kwargs[key] = parsed_value
                    else:
                        fixed_kwargs[key] = value
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse string dict for key '{key}': {value}, error: {e}")
                    # Try a more lenient JSON parsing
                    try:
                        import json
                        # Replace single quotes with double quotes for JSON compatibility
                        json_str = value.replace("'", '"')
                        parsed_value = json.loads(json_str)
                        logger.debug(f"Converted string dict using JSON for key '{key}': {value} -> {parsed_value}")
                        fixed_kwargs[key] = parsed_value
                    except:
                        # If all parsing fails, keep the original value
                        logger.warning(f"Could not parse string dict for key '{key}', keeping as string")
                        fixed_kwargs[key] = value
            else:
                fixed_kwargs[key] = value
        
        return fixed_kwargs

    async def _arun(self, **kwargs) -> str:
        """Execute the MCP tool asynchronously"""
        try:
            # Fix string dicts for async execution too
            kwargs = self._fix_string_dicts(kwargs)
            connection = await MCPConnectionManager.get_connection(tuple(self.server_command))
            return await connection.call_tool(self.tool_name, kwargs)
        except Exception as e:
            return f"Execution error: {str(e)}"


class MCPConnectionManager:
    """Global manager for MCP server connections"""
    _connections = {}
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_connection(cls, server_command_tuple):
        """Get or create a connection for a server command"""
        async with cls._lock:
            if server_command_tuple not in cls._connections:
                cls._connections[server_command_tuple] = MCPServerConnection(list(server_command_tuple))
            return cls._connections[server_command_tuple]
    
    @classmethod
    async def close_all(cls):
        """Close all connections"""
        async with cls._lock:
            for connection in cls._connections.values():
                await connection.close()
            cls._connections.clear()


class MCPServerConnection:
    """Manages a persistent connection to an MCP server"""
    
    def __init__(self, server_command):
        self.server_command = server_command
        self.process = None
        self.initialized = False
        self._lock = asyncio.Lock()
    
    async def _ensure_connected(self):
        """Ensure the server is running and initialized"""
        if self.process is None or self.process.returncode is not None:
            await self._start_server()
        
        if not self.initialized:
            await self._initialize_server()
    
    async def _start_server(self):
        """Start the MCP server process"""
        try:
            logger.info(f"Starting MCP server: {' '.join(self.server_command)}")
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()  # Ensure correct working directory
            )
            logger.info(f"MCP server started with PID: {self.process.pid}")
            self.initialized = False
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise Exception(f"Failed to start MCP server: {e}")
    
    async def _initialize_server(self):
        """Initialize the MCP server"""
        try:
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "streamlined-consciousness", "version": "1.0.0"}
                }
            }
            
            await self._send_message(init_request)
            
            # Read initialization response
            response = await self._read_message()
            
            if "error" in response:
                raise Exception(f"Initialization error: {response['error']}")
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            await self._send_message(initialized_notification)
            self.initialized = True
            
        except Exception as e:
            raise Exception(f"Failed to initialize MCP server: {e}")
    
    async def _send_message(self, message):
        """Send a JSON-RPC message to the server"""
        if self.process is None:
            raise Exception("Server process not started")
        
        message_str = json.dumps(message) + "\n"
        self.process.stdin.write(message_str.encode())
        await self.process.stdin.drain()
    
    async def _read_message(self):
        """Read a JSON-RPC message from the server"""
        if self.process is None:
            raise Exception("Server process not started")
        
        try:
            # Increased timeout to allow for model loading
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=180.0)
            if not line:
                # Check if process has terminated and capture stderr
                if self.process.returncode is not None:
                    stderr_output = await self.process.stderr.read()
                    stderr_text = stderr_output.decode() if stderr_output else "No stderr output"
                    logger.error(f"Server terminated with code {self.process.returncode}. Stderr: {stderr_text}")
                    raise Exception(f"Server closed connection (exit code: {self.process.returncode}, stderr: {stderr_text})")
                else:
                    raise Exception("Server closed connection")
            
            return json.loads(line.decode().strip())
        except asyncio.TimeoutError:
            logger.error("Server response timeout - checking if process is still alive")
            if self.process.returncode is not None:
                stderr_output = await self.process.stderr.read()
                stderr_text = stderr_output.decode() if stderr_output else "No stderr output"
                logger.error(f"Server died during timeout. Exit code: {self.process.returncode}, Stderr: {stderr_text}")
            raise Exception("Server response timeout")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise Exception(f"Invalid JSON response: {e}")
    
    async def call_tool(self, tool_name, arguments):
        """Call a tool on the MCP server"""
        async with self._lock:
            await self._ensure_connected()
            
            # Call the tool
            tool_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            await self._send_message(tool_request)
            
            # Read tool response
            response = await self._read_message()
            
            if "error" in response:
                return f"Tool error: {response['error']}"
            
            # Extract content
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", str(content))
                return str(content)
            
            return str(response.get("result", "No result"))
    
    async def close(self):
        """Close the server connection"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None
        self.initialized = False

# Argument schemas for tools
class SimpleArgs(BaseModel):
    """Simple arguments for tools that don't need complex parameters"""
    pass

class ConceptArgs(BaseModel):
    name: str = Field(description="Name of the concept")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Additional properties (optional)")
    
    @field_validator('properties', mode='before')
    @classmethod
    def convert_string_dict(cls, v):
        """Convert string representation of dict to actual dict for Gemini compatibility"""
        if v is None:
            return v
        if isinstance(v, str) and v.strip().startswith('{') and v.strip().endswith('}'):
            import ast
            import json
            try:
                # Try ast.literal_eval first (safer)
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                try:
                    # Fallback to JSON parsing with single quote conversion
                    json_str = v.replace("'", '"')
                    return json.loads(json_str)
                except:
                    # If all parsing fails, return empty dict
                    logger.warning(f"Could not parse string dict in ConceptArgs: {v}")
                    return {}
        return v

class RelationshipArgs(BaseModel):
    from_concept: str = Field(description="Source concept")
    to_concept: str = Field(description="Target concept")
    relationship_type: str = Field(description="Type of relationship")
    properties: Dict[str, Any] = Field(default={}, description="Additional properties")
    
    @field_validator('properties', mode='before')
    @classmethod
    def convert_string_dict(cls, v):
        """Convert string representation of dict to actual dict for Gemini compatibility"""
        if isinstance(v, str) and v.strip().startswith('{') and v.strip().endswith('}'):
            import ast
            import json
            try:
                # Try ast.literal_eval first (safer)
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                try:
                    # Fallback to JSON parsing with single quote conversion
                    json_str = v.replace("'", '"')
                    return json.loads(json_str)
                except:
                    # If all parsing fails, return empty dict
                    logger.warning(f"Could not parse string dict in RelationshipArgs: {v}")
                    return {}
        return v

class HyperedgeArgs(BaseModel):
    members: List[str] = Field(description="Concepts to connect")
    label: Optional[str] = Field(default=None, description="Optional label")

class QueryArgs(BaseModel):
    query: str = Field(description="Cypher query to execute")
    parameters: Dict[str, Any] = Field(default={}, description="Query parameters")
    
    @field_validator('parameters', mode='before')
    @classmethod
    def convert_string_dict(cls, v):
        """Convert string representation of dict to actual dict for Gemini compatibility"""
        if isinstance(v, str) and v.strip().startswith('{') and v.strip().endswith('}'):
            import ast
            import json
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                try:
                    json_str = v.replace("'", '"')
                    return json.loads(json_str)
                except:
                    logger.warning(f"Could not parse string dict in QueryArgs: {v}")
                    return {}
        return v

class ConceptExploreArgs(BaseModel):
    concept_name: str = Field(description="Name of the concept to explore")
    max_depth: int = Field(default=2, description="Maximum depth to explore")

class MemoryStoreArgs(BaseModel):
    content: str = Field(description="Content to store")
    vector: List[float] = Field(description="Vector embedding (384 dimensions)")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    @field_validator('metadata', mode='before')
    @classmethod
    def convert_string_dict(cls, v):
        """Convert string representation of dict to actual dict for Gemini compatibility"""
        if isinstance(v, str) and v.strip().startswith('{') and v.strip().endswith('}'):
            import ast
            import json
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                try:
                    json_str = v.replace("'", '"')
                    return json.loads(json_str)
                except:
                    logger.warning(f"Could not parse string dict in MemoryStoreArgs: {v}")
                    return {}
        return v

class MemorySearchArgs(BaseModel):
    query_vector: List[float] = Field(description="Query vector for similarity search")
    limit: int = Field(default=10, description="Maximum results")
    score_threshold: float = Field(default=0.0, description="Minimum similarity score")

class EmbeddingArgs(BaseModel):
    text: str = Field(description="Text to generate embedding for")
    normalize: bool = Field(default=True, description="Whether to normalize the embedding")

class SimilarityArgs(BaseModel):
    text1: str = Field(description="First text")
    text2: str = Field(description="Second text")
    metric: str = Field(default="cosine", description="Similarity metric")

class ProjectArgs(BaseModel):
    description: str = Field(description="Project description")
    name: str = Field(default="", description="Optional project name")

class IterationArgs(BaseModel):
    user_input: str = Field(description="User's request for a dream session")

class NukeArgs(BaseModel):
    confirm: bool = Field(description="Confirmation flag")

class NukeProjectArgs(BaseModel):
    name: str = Field(description="Name of the project to delete")
    confirm: bool = Field(description="Confirmation flag")

class NukeQdrantArgs(BaseModel):
    name: str = Field(description="Name of the collection to delete")
    confirm: bool = Field(description="Confirmation flag")

class TextMemoryStoreArgs(BaseModel):
    text: str = Field(description="Text content to store")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    @field_validator('metadata', mode='before')
    @classmethod
    def convert_string_dict(cls, v):
        """Convert string representation of dict to actual dict for Gemini compatibility"""
        if isinstance(v, str) and v.strip().startswith('{') and v.strip().endswith('}'):
            import ast
            import json
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                try:
                    json_str = v.replace("'", '"')
                    return json.loads(json_str)
                except:
                    logger.warning(f"Could not parse string dict in TextMemoryStoreArgs: {v}")
                    return {}
        return v

class TextMemorySearchArgs(BaseModel):
    query_text: str = Field(description="Text to search for")
    limit: int = Field(default=5, description="Maximum number of results")
    score_threshold: float = Field(default=0.0, description="Minimum similarity score")

class EmbedDocumentArgs(BaseModel):
    file_path: str = Field(description="Path to file relative to data folder")
    chunk_size: int = Field(default=1000, description="Max chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata to store")
    collection_name: str = Field(default="documents", description="Collection to store in")

class ListEmbeddedDocumentsArgs(BaseModel):
    collection_name: str = Field(default="documents", description="Collection to list from")

class DeleteDocumentEmbeddingsArgs(BaseModel):
    file_path: str = Field(description="Path to file that was embedded")
    collection_name: str = Field(default="documents", description="Collection to delete from")

class SearchDocumentsArgs(BaseModel):
    query_text: str = Field(description="Text to search for")
    limit: int = Field(default=5, description="Maximum number of results")
    file_filter: Optional[str] = Field(default=None, description="Filter by source file")
    collection_name: str = Field(default="documents", description="Collection to search in")

class HausdorffArgs(BaseModel):
    dimension_method: str = Field(default="box_counting", description="Method to use: box_counting, correlation, information, sandbox, mass_radius, local_pointwise, multifractal")
    weight_threshold: float = Field(default=0.3, description="Minimum edge weight to consider")
    scale_range: List[float] = Field(default=[0.1, 2.0, 20], description="Scale range [min, max, num_scales]")
    project_id: str = Field(default="default", description="Project ID to analyze")
    store_history: bool = Field(default=True, description="Store calculation in metadata")

class BatchSimilaritiesArgs(BaseModel):
    concept_pairs: List[List[str]] = Field(description="List of concept pairs [[c1,c2], ...]")
    boost_hyperedge_members: bool = Field(default=True, description="Boost for hyperedge co-membership")

def create_utility_tools() -> List[StreamlinedMCPTool]:
    """Create utility tools"""
    # Create a proper MCP server for utility functions
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create inline utility server command
    utility_server_script = f'''
import asyncio
import json
import re
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, ServerCapabilities

def extract_iterations_from_text(user_input: str) -> int:
    """Extract number of iterations from user input"""
    if not user_input:
        return 3
    
    user_lower = user_input.lower()
    
    # Look for number patterns
    patterns = [
        r'(\\d+)\\s*iterations?',
        r'(\\d+)\\s*times?', 
        r'for\\s+(\\d+)',
        r'dream\\s+(\\d+)',
        r'(\\d+)\\s*turns?',
        r'(\\d+)\\s*rounds?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_lower)
        if match:
            try:
                num = int(match.group(1))
                if num >= 1:  # Only ensure it's at least 1
                    return num
            except ValueError:
                continue
    
    return 3  # Default

server = Server("utility")

@server.list_tools()
async def handle_list_tools():
    return [
        Tool(
            name="extract_dream_iterations",
            description="Extract number of dream iterations from user input",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "user_input": {{"type": "string", "description": "User's dream request"}}
                }},
                "required": ["user_input"]
            }}
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "extract_dream_iterations":
        user_input = arguments.get("user_input", "")
        iterations = extract_iterations_from_text(user_input)
        return [TextContent(type="text", text=str(iterations))]
    else:
        raise ValueError(f"Unknown tool: {{name}}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="utility",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools={{}})
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Write utility server to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=current_dir) as f:
        f.write(utility_server_script)
        utility_server_path = f.name
    
    return [
        StreamlinedMCPTool(
            server_name="utility",
            tool_name="extract_dream_iterations",
            description="Extracts the number of dream iterations from a user's request. Defaults to 3 if no number is specified.",
            args_schema=IterationArgs,
            server_command=["python", utility_server_path]
        )
    ]

def create_neo4j_core_tools() -> List[StreamlinedMCPTool]:
    """Create core Neo4j tools for basic graph operations"""
    # Get absolute path to the server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(current_dir), "mcp_servers", "neo4j_hypergraph", "server.py")
    server_command = ["python", "-u", server_path]
    
    return [
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="create_concept_node",
            description="Create a new concept node in the semantic hypergraph",
            args_schema=ConceptArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="create_relationship",
            description="Create a weighted semantic relationship between concepts",
            args_schema=RelationshipArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="get_concept_neighbors",
            description="Get concepts connected to a given concept",
            args_schema=ConceptExploreArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="get_graph_stats",
            description="Get statistics about the current hypergraph",
            args_schema=SimpleArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="query_cypher",
            description="Execute a Cypher query on the hypergraph",
            args_schema=QueryArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="calculate_hausdorff_dimension",
            description="Calculate the fractal dimension of the semantic hypergraph. Available methods: box_counting (default), correlation, information, sandbox, mass_radius, local_pointwise, multifractal. Use dimension_method parameter to specify method.",
            args_schema=HausdorffArgs,
            server_command=server_command
        )
    ]

def create_neo4j_evolution_tools() -> List[StreamlinedMCPTool]:
    """Create Neo4j tools for knowledge evolution - INTERNAL USE ONLY (not exposed to LLM)"""
    # Get absolute path to the server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(current_dir), "mcp_servers", "neo4j_hypergraph", "server.py")
    server_command = ["python", server_path]
    
    # These tools exist for internal consciousness engine use during dream coordination
    # They should NEVER be selected during _select_contextual_tools() for LLM use
    return [
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="apply_ca_rules_enhanced",
            description="Enhanced CA with pruning and noise-adding for dream states - INTERNAL ONLY",
            args_schema=SimpleArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="create_hyperedge",
            description="Create a semantic hyperedge connecting multiple concepts - INTERNAL ONLY",
            args_schema=HyperedgeArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="get_ca_connection_candidates",
            description="Get connection candidates for CA without truncation - INTERNAL ONLY",
            args_schema=SimpleArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="calculate_batch_similarities",
            description="Calculate batch semantic similarities - INTERNAL ONLY",
            args_schema=BatchSimilaritiesArgs,
            server_command=server_command
        )
    ]

def create_neo4j_project_tools() -> List[StreamlinedMCPTool]:
    """Create Neo4j tools for project management"""
    # Get absolute path to the server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(current_dir), "mcp_servers", "neo4j_hypergraph", "server.py")
    server_command = ["python", server_path]
    
    return [
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="create_blank_project",
            description="Create a new blank consciousness project",
            args_schema=ProjectArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="list_my_projects",
            description="List all available consciousness projects",
            args_schema=SimpleArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="switch_to_project",
            description="Switch to a different consciousness project",
            args_schema=ProjectArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="describe_current_project",
            description="Get information about the current project",
            args_schema=SimpleArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="delete_project",
            description="Delete a project and all its data",
            args_schema=NukeProjectArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="neo4j-hypergraph",
            tool_name="archive_project",
            description="Archive a consciousness project (mark as inactive)",
            args_schema=ProjectArgs,
            server_command=server_command
        )
    ]

def create_qdrant_memory_tools() -> List[StreamlinedMCPTool]:
    """Create Qdrant memory tools"""
    # Get absolute path to the server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(current_dir), "mcp_servers", "qdrant_memory", "server.py")
    server_command = ["python", server_path]
    
    return [
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="store_text_memory",
            description="Store a text memory (embedding generated internally - no vectors sent to LLM)",
            args_schema=TextMemoryStoreArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="search_text_memories",
            description="Search memories using text query (embedding generated internally - no vectors sent to LLM)",
            args_schema=TextMemorySearchArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="get_collection_info",
            description="Get information about the memory collection",
            args_schema=SimpleArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="embed_document",
            description="Embed a document from the data folder into vector memory",
            args_schema=EmbedDocumentArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="list_embedded_documents",
            description="List all documents that have been embedded",
            args_schema=ListEmbeddedDocumentsArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="delete_document_embeddings",
            description="Delete all embeddings from a specific document",
            args_schema=DeleteDocumentEmbeddingsArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="qdrant-memory",
            tool_name="search_documents",
            description="Search through embedded documents",
            args_schema=SearchDocumentsArgs,
            server_command=server_command
        )
    ]

def create_sentence_transformer_tools() -> List[StreamlinedMCPTool]:
    """Create sentence transformer tools - optimized to avoid token waste"""
    # Get absolute path to the server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(os.path.dirname(current_dir), "mcp_servers", "sentence_transformers", "server.py")
    server_command = ["python", server_path]
    
    return [
        # Removed generate_embedding - it wastes tokens with 384-dimensional arrays
        # Memory functions handle embedding generation internally
        StreamlinedMCPTool(
            server_name="sentence-transformers",
            tool_name="similarity_score",
            description="Calculate semantic similarity between two texts",
            args_schema=SimilarityArgs,
            server_command=server_command
        ),
        StreamlinedMCPTool(
            server_name="sentence-transformers",
            tool_name="get_model_info",
            description="Get information about the sentence transformer model",
            args_schema=SimpleArgs,
            server_command=server_command
        )
    ]


def create_consciousness_tools(consciousness_engine) -> List[BaseTool]:
    """Create consciousness-specific tools that need access to the engine"""
    return [
        DreamTool(consciousness_engine)
    ]

def create_tool_categories(consciousness_engine=None) -> Dict[str, ToolCategory]:
    """Create organized tool categories for intelligent loading"""
    
    categories = {
        "consciousness": ToolCategory(
            name="consciousness",
            description="Core consciousness functions like dreaming",
            tools=create_consciousness_tools(consciousness_engine) if consciousness_engine else [],
            priority=12,  # Highest priority
            always_available=True
        ),
        
        "utility": ToolCategory(
            name="utility",
            description="General purpose utility tools",
            tools=create_utility_tools(),
            priority=11,
            always_available=True
        ),

        "neo4j-core": ToolCategory(
            name="neo4j-core",
            description="Core hypergraph operations - concepts, relationships, queries",
            tools=create_neo4j_core_tools(),
            priority=10,  # Highest priority
            always_available=True  # Always include these
        ),
        
        "sentence-transformers": ToolCategory(
            name="sentence-transformers",
            description="Embedding generation and semantic similarity",
            tools=create_sentence_transformer_tools(),
            priority=9,  # Very high priority (needed for memory)
            always_available=False
        ),
        
        "qdrant-memory": ToolCategory(
            name="qdrant-memory",
            description="Long-term vector memory storage and retrieval",
            tools=create_qdrant_memory_tools(),
            priority=8,  # High priority
            always_available=False
        ),
        
        "neo4j-evolution": ToolCategory(
            name="neo4j-evolution",
            description="Knowledge evolution and cellular automata",
            tools=create_neo4j_evolution_tools(),
            priority=6,  # Medium priority
            always_available=False
        ),
        
        "neo4j-projects": ToolCategory(
            name="neo4j-projects",
            description="Project and workspace management",
            tools=create_neo4j_project_tools(),
            priority=7,  # Raised priority - these are important tools
            always_available=True  # Always available since Elder needs project management
        )
    }
    
    return categories

def register_all_tools(consciousness_engine):
    """Register all tool categories with the consciousness engine"""
    categories = create_tool_categories(consciousness_engine)
    
    for category_name, category in categories.items():
        consciousness_engine.register_tool_category(category)
    
    total_tools = sum(len(cat.tools) for cat in categories.values())
    logger.info(f"✅ Registered {len(categories)} tool categories with {total_tools} total tools")
    
    return categories

async def test_tool_creation():
    """Test tool creation"""
    categories = create_tool_categories()
    
    print("🛠️ Streamlined Tool Categories:")
    for name, category in categories.items():
        print(f"  📦 {name}: {len(category.tools)} tools (priority: {category.priority})")
        for tool in category.tools:
            print(f"    - {tool.name}")

if __name__ == "__main__":
    asyncio.run(test_tool_creation())
