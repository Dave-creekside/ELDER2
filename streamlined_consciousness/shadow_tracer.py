"""
Shadow Tracer Module (Real-World)
Captures actual internal hidden states of the Student model using PyTorch hooks.
"""

import logging
import asyncio
import uuid
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from mcp_servers.qdrant_memory.server import QdrantMemoryServer

# Configure logging
logger = logging.getLogger("shadow-tracer")

class ShadowTracer:
    """
    Hooks into the local Student Model to capture internal activation traces.
    """
    
    def __init__(self, student_model, qdrant_server: Optional[QdrantMemoryServer] = None):
        self.student = student_model
        self.qdrant = qdrant_server
        self.collection_name = "shadow_traces"
        self.buffer: List[Dict[str, Any]] = []
        self.current_trace = None
        self.hooks = []
        
        # Determine hidden dimension from model config if loaded
        self.embedding_dim = 3072 # Default for Gemma-4b, will update on load
        
    def register_hooks(self):
        """Register forward hooks on the model to capture hidden states"""
        if not self.student.model:
            logger.warning("Cannot register hooks: Student model not loaded")
            return

        # Update dim from config
        self.embedding_dim = self.student.get_hidden_size()
        logger.info(f"Shadow Tracer initialized for hidden dim: {self.embedding_dim}")

        # Hook into the last transformer layer
        # For Gemma/Llama, usually model.model.layers[-1] or model.layers[-1]
        try:
            # Try to find the layers attribute robustly
            target_layer = None
            
            # Common paths for transformer layers
            paths = [
                lambda m: m.model.layers,      # Standard Llama/Gemma/Mistral
                lambda m: m.layers,            # Some configurations
                lambda m: m.transformer.h,     # GPT-2/Phi-2 style
                lambda m: m.base_model.model.model.layers, # PEFT wrapped
                lambda m: m.base_model.model.layers,      # PEFT wrapped alternate
                # Multimodal/VLM specific paths
                lambda m: m.language_model.model.layers,
                lambda m: m.text_model.encoder.layers,
            ]
            
            for i, path_fn in enumerate(paths):
                try:
                    layers = path_fn(self.student.model)
                    if layers and len(layers) > 0:
                        target_layer = layers[-1]
                        logger.info(f"✅ Found transformer layers using path strategy #{i+1}")
                        break
                except:
                    continue
            
            if target_layer is None:
                # Fallback: iterate through named modules to find layers
                logger.info("Searching for layers in named modules...")
                for name, module in self.student.model.named_modules():
                    # Skip vision/audio towers if possible
                    if 'vision' in name or 'audio' in name:
                        continue
                        
                    if 'layers' in name.split('.') and hasattr(module, '__len__') and len(module) > 0:
                        target_layer = module[-1]
                        logger.info(f"✅ Found target layer in module: {name}")
                        break

            if target_layer is None:
                raise AttributeError("Could not find transformer layers in model structure")
                
            def hook_fn(module, input, output):
                # Output is usually a tuple (hidden_states, ...)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Capture the last token's hidden state (representing the 'thought' at that step)
                # Or average over the sequence. For simplicity, we take the mean of the sequence.
                # Shape: [batch, seq_len, hidden_dim]
                
                with torch.no_grad():
                    # Move to CPU and numpy
                    trace_vector = hidden_states.mean(dim=1).float().cpu().numpy()
                    
                    # Store for retrieval
                    self.current_trace = trace_vector[0].tolist() # Batch size 1 assumption
            
            handle = target_layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
            logger.info("✅ Shadow hook registered on last transformer layer")
            
        except Exception as e:
            logger.error(f"Failed to register hook: {e}")

    async def _ensure_qdrant(self):
        """Ensure Qdrant connection is available"""
        if self.qdrant is None:
            try:
                self.qdrant = QdrantMemoryServer()
                await self.qdrant.connect_qdrant()
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")

    async def capture_trace(self, input_text: str, output_text: str, 
                          anchor_node: str, state: str = "wake"):
        """
        Run the Student model on the input/output pair and capture the trace.
        """
        if not self.student.model:
            logger.warning("Student model not loaded, skipping trace")
            return

        try:
            # We want to trace the *process* of generating the output from the input.
            # So we feed "Input + Output" to the student and see how it represents it.
            full_prompt = f"{input_text}\n{output_text}"
            
            # Reset current trace
            self.current_trace = None
            
            # Run forward pass (just encoding/generating to trigger hook)
            # We don't need to generate new text, just process the context
            inputs = self.student.tokenizer(full_prompt, return_tensors="pt").to(self.student.device)
            
            with torch.no_grad():
                self.student.model(**inputs)
            
            if self.current_trace:
                # We have a vector!
                
                # Calculate Input Mean for Hebbian pairing (x)
                # We can approximate this by running just the input
                input_inputs = self.student.tokenizer(input_text, return_tensors="pt").to(self.student.device)
                with torch.no_grad():
                    # Temporarily replace current_trace with input trace
                    temp_trace = self.current_trace
                    self.current_trace = None
                    self.student.model(**input_inputs)
                    input_mean = self.current_trace
                    self.current_trace = temp_trace # Restore output trace (which is actually the full sequence trace)
                
                # If we couldn't get input mean, use zeros (fallback)
                if not input_mean:
                    input_mean = [0.0] * self.embedding_dim

                # Construct payload
                trace_id = str(uuid.uuid4())
                trace_payload = {
                    "id": trace_id,
                    "vector": self.current_trace, # The actual hidden state trace
                    "payload": {
                        "anchor_node": anchor_node,
                        "state": state,
                        "input_text": input_text[:200],
                        "input_mean": input_mean,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                }
                
                self.buffer.append(trace_payload)
                logger.info(f"Captured REAL shadow trace (dim={len(self.current_trace)}) for '{anchor_node}'")
                
                # Flush immediately for development responsiveness
                if len(self.buffer) >= 1:
                    await self.flush_traces()
            else:
                logger.warning("Forward pass completed but no trace captured (hook failed?)")

        except Exception as e:
            logger.error(f"Error capturing trace: {e}")

    async def flush_traces(self):
        """Flush buffered traces to Qdrant"""
        if not self.buffer:
            return
            
        await self._ensure_qdrant()
        if not self.qdrant or not self.qdrant.client:
            return
            
        try:
            from qdrant_client.models import PointStruct, VectorParams, Distance
            
            # Verify collection dimension matches current traces
            vector_dim = len(self.buffer[0]["vector"])
            try:
                info = await self.qdrant.client.get_collection(self.collection_name)
                current_dim = info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else 0
                if current_dim != vector_dim:
                    logger.warning(f"⚠️ Dimension mismatch in Qdrant: {current_dim} vs {vector_dim}. Recreating...")
                    await self.qdrant.client.delete_collection(self.collection_name)
                    raise Exception("Collection recreate needed") # Jump to creation
            except:
                # Create collection if missing or just deleted
                await self.qdrant.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
                )
                logger.info(f"✅ Re-initialized '{self.collection_name}' with {vector_dim} dimensions")

            points = []
            for trace in self.buffer:
                # Ensure vector is a flat list of floats and matches expected dimension
                vec = [float(x) for x in trace["vector"]]
                if len(vec) != vector_dim:
                    logger.error(f"❌ Vector dimension mismatch: {len(vec)} vs {vector_dim}")
                    continue
                
                # Diagnostic logging
                logger.info(f"Preparing trace point {trace['id']} with vector sample: {vec[:3]}...")

                # Use dictionary format instead of PointStruct for better compatibility
                points.append({
                    "id": trace["id"],
                    "vector": vec,
                    "payload": trace["payload"]
                })
            
            if not points:
                logger.warning("No valid points to flush")
                return

            # Explicit dictionary-based upsert
            await self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Flushed {len(points)} traces to {self.collection_name}")
            self.buffer = []
                
        except Exception as e:
            logger.error(f"Failed to flush traces: {e}")

    def remove_hooks(self):
        """Cleanup hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
