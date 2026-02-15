"""
Deep Sleep Engine (Real-World)
Implements Riemannian consolidation of shadow traces into PEFT LoRA weight updates.
"""

import logging
import asyncio
import numpy as np
import torch
import os
import sys
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.qdrant_memory.server import QdrantMemoryServer
from mcp_servers.neo4j_hypergraph.server import Neo4jSemanticHypergraphServer
from streamlined_consciousness.student_model import StudentModel
from streamlined_consciousness.config import config

# Configure logging
logger = logging.getLogger("deep-sleep-engine")

class DeepSleepEngine:
    def __init__(self, qdrant_server: Optional[QdrantMemoryServer] = None, 
                 neo4j_server: Optional[Neo4jSemanticHypergraphServer] = None,
                 student_model: Optional[StudentModel] = None):
        self.qdrant = qdrant_server
        self.neo4j = neo4j_server
        self.student = student_model
        self.trace_collection = "shadow_traces"
        self.embedding_dim = 3072 # Will update from model
        self.k_components = config.DEEP_SLEEP_SVD_RANK
        self.learning_rate = 0.001

    async def _ensure_connections(self):
        """Ensure database connections are available"""
        if self.qdrant is None:
            try:
                self.qdrant = QdrantMemoryServer()
                await self.qdrant.connect_qdrant()
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
        
        if self.neo4j is None:
            try:
                self.neo4j = Neo4jSemanticHypergraphServer()
                await self.neo4j.connect_neo4j()
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")

    async def _load_student(self):
        """Load student model and adapter if not already provided"""
        if self.student is None:
            try:
                logger.info("Loading Student Model for Deep Sleep...")
                self.student = StudentModel()
                self.student.load()
                self.embedding_dim = self.student.get_hidden_size()
            except Exception as e:
                logger.error(f"Failed to load student model: {e}")
        elif self.student.model is None:
            # Model instance exists but weights aren't loaded
            try:
                self.student.load()
                self.embedding_dim = self.student.get_hidden_size()
            except Exception as e:
                logger.error(f"Failed to load existing student model: {e}")
        else:
            # Already loaded
            self.embedding_dim = self.student.get_hidden_size()

    async def perform_deep_sleep_cycle(self):
        """
        Execute the full Deep Sleep consolidation cycle.
        1. Fetch traces.
        2. Calculate Geometric Updates (SVD + Natural Gradient).
        3. Apply updates to PEFT adapter weights.
        4. Save adapter.
        """
        await self._ensure_connections()
        await self._load_student()
        
        if not self.qdrant or not self.student:
            logger.error("Cannot perform deep sleep: Missing dependencies")
            return

        logger.info("💤 Starting Deep Sleep Cycle (Real-World Mode)...")
        
        try:
            # 1. Fetch all traces
            result = await self.qdrant.client.scroll(
                collection_name=self.trace_collection,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            points, _ = result
            
            if not points:
                logger.info("No traces to process. Waking up.")
                return

            logger.info(f"Processing {len(points)} traces...")
            
            # Group by Anchor Node
            traces_by_concept = {}
            for point in points:
                anchor = point.payload.get("anchor_node", "unknown")
                if anchor not in traces_by_concept:
                    traces_by_concept[anchor] = []
                traces_by_concept[anchor].append(point)
            
            total_updates_magnitude = 0.0
            
            # We will accumulate gradients for LoRA matrices A and B
            # LoRA update: W += B * A
            # Our Hebbian update is conceptually on W directly: dW = natural_delta * input^T
            # We need to project dW into dA and dB space.
            # Simplified approach: Update B (output projection) to align with natural_delta
            # while keeping A (input projection) fixed or updating it to align with input.
            
            # For this implementation, we'll apply a simplified update to lora_B
            # d_lora_B += lr * (natural_delta * (lora_A * input))
            
            # Identify target modules (e.g., q_proj, v_proj)
            target_modules = config.LORA_TARGET_MODULES
            
            total_modules_updated = 0
            total_modules_skipped = 0
            concepts_processed = 0
            
            for concept, concept_traces in traces_by_concept.items():
                if len(concept_traces) < 2:
                    logger.info(f"Skipping concept '{concept}': only {len(concept_traces)} trace(s) (need ≥2)")
                    continue 
                
                concepts_processed += 1
                concept_updates = 0
                concept_skips = 0
                logger.info(f"📐 Consolidating geometry for concept: {concept} ({len(concept_traces)} traces)")
                
                # Extract vectors (deltas)
                deltas = np.array([t.vector for t in concept_traces])
                
                # SVD for Metric Tensor
                try:
                    U, S, Vt = np.linalg.svd(deltas.T, full_matrices=False)
                    
                    # Adaptive Rank Selection
                    if self.k_components <= 0:
                        # Adaptive: Keep 95% of explained variance
                        total_variance = np.sum(S**2)
                        if total_variance > 1e-9:
                            cumulative_variance = np.cumsum(S**2)
                            # Find index where variance >= 0.95
                            explained_ratio = cumulative_variance / total_variance
                            k = np.searchsorted(explained_ratio, 0.95) + 1
                            logger.info(f"Adaptive SVD: Selected k={k} (95% variance) for {concept}")
                        else:
                            k = 1
                    else:
                        k = min(self.k_components, len(S))
                        
                    U_k = U[:, :k]
                    S_k = S[:k]
                    
                    # Update Neo4j
                    await self._update_concept_geometry(concept, U_k, S_k)
                    
                    # Process traces
                    for i, point in enumerate(concept_traces):
                        delta = deltas[i]
                        input_mean = np.array(point.payload.get("input_mean", [0.0]*self.embedding_dim))
                        
                        # Natural Gradient Calculation
                        projections = U_k.T @ delta
                        damped_projections = projections / (S_k + 1e-6)
                        natural_delta = delta - (U_k @ (projections - damped_projections))
                        
                        # Apply to LoRA weights
                        # Iterate through all LoRA layers and apply a small update
                        # This spreads the knowledge across the network (holographic-like)
                        # In a more advanced version, we'd use attribution to select specific layers
                        
                        with torch.no_grad():
                            for name, module in self.student.model.named_modules():
                                if any(t in name for t in target_modules) and hasattr(module, "lora_B"):
                                    # module.lora_B is [out_features, rank]
                                    # module.lora_A is [rank, in_features]
                                    
                                    # Detect target dtype and device from the LoRA weights
                                    # (usually float32 even if model is 4bit)
                                    # Use the module's actual device to handle multi-GPU setups
                                    target_dtype = module.lora_A['default'].weight.dtype
                                    module_device = module.lora_A['default'].weight.device
                                    
                                    # Get each module's actual dimensions
                                    # GQA models have asymmetric projections (e.g., q_proj: in=2560, out=2048)
                                    lora_a_in_dim = module.lora_A['default'].weight.shape[1]   # [rank, in_features]
                                    lora_b_out_dim = module.lora_B['default'].weight.shape[0]   # [out_features, rank]
                                    
                                    # Adapt trace vectors to match this module's dimensions
                                    # Truncate if trace is longer, zero-pad if shorter
                                    def adapt_vector(vec, target_dim):
                                        """Adapt a numpy vector to target dimension by truncating or zero-padding"""
                                        if len(vec) == target_dim:
                                            return vec
                                        elif len(vec) > target_dim:
                                            return vec[:target_dim]  # Truncate to leading components
                                        else:
                                            return np.pad(vec, (0, target_dim - len(vec)))  # Zero-pad
                                    
                                    # Adapt input_mean to match lora_A's in_features
                                    adapted_input = adapt_vector(input_mean, lora_a_in_dim)
                                    # Adapt natural_delta to match lora_B's out_features
                                    adapted_delta = adapt_vector(natural_delta, lora_b_out_dim)
                                    
                                    # Convert to torch tensors on the same device as this module's weights
                                    natural_delta_t = torch.tensor(adapted_delta, device=module_device, dtype=target_dtype)
                                    input_mean_t = torch.tensor(adapted_input, device=module_device, dtype=target_dtype)
                                    
                                    # Hebbian update: dB ~ natural_delta * (A*x)^T
                                    # Project input through A: [rank, in] @ [in] -> [rank]
                                    mid_act = module.lora_A['default'].weight @ input_mean_t
                                    
                                    # Outer product: [out] x [rank] -> [out, rank]
                                    update = torch.outer(natural_delta_t, mid_act)
                                    
                                    # Scale by learning rate and LoRA scaling
                                    scaling = module.scaling['default']
                                    update = update * self.learning_rate / scaling
                                    
                                    # Apply update
                                    module.lora_B['default'].weight += update
                                    total_updates_magnitude += update.norm().item()
                                    concept_updates += 1
                                    total_modules_updated += 1

                    logger.info(f"   ✅ Concept '{concept}': {concept_updates} LoRA module updates applied across all layers")

                except Exception as e:
                    logger.error(f"Error processing concept {concept}: {e}")
                    continue

            # === TRAINING SUMMARY ===
            logger.info("=" * 60)
            logger.info("📊 DEEP SLEEP TRAINING SUMMARY")
            logger.info(f"   Traces processed:      {len(points)}")
            logger.info(f"   Concepts consolidated:  {concepts_processed} / {len(traces_by_concept)}")
            logger.info(f"   LoRA modules updated:   {total_modules_updated}")
            logger.info(f"   Total update magnitude: {total_updates_magnitude:.6f}")
            logger.info(f"   SVD rank (k):           {self.k_components}")
            logger.info(f"   Learning rate:          {self.learning_rate}")
            logger.info(f"   Hidden dim:             {self.embedding_dim}")
            logger.info("=" * 60)
            
            # Save updated adapter
            if total_updates_magnitude > 0:
                logger.info(f"💾 Saving updated adapter (Total Magnitude: {total_updates_magnitude:.4f})")
                self.student.save_adapter()
                logger.info(f"   Adapter saved to: {self.student.adapter_path}")
            else:
                logger.warning("⚠️ No weight updates were applied — adapter NOT saved")
            
            # Cleanup traces
            points_ids = [p.id for p in points]
            await self.qdrant.client.delete(
                collection_name=self.trace_collection,
                points_selector=points_ids
            )
            
            logger.info("✅ Deep Sleep Cycle Complete")

        except Exception as e:
            logger.error(f"Deep sleep cycle failed: {e}")

    async def _update_concept_geometry(self, concept_name: str, U: np.ndarray, S: np.ndarray):
        """Update Neo4j concept node with new geometric basis"""
        if not self.neo4j:
            return
        
        singular_values = S.tolist()
        
        async with self.neo4j.driver.session() as session:
            query = """
            MATCH (n:Concept {name: $name})
            SET n.singular_values = $s,
                n.last_sleep_update = datetime()
            RETURN n
            """
            await session.run(query, name=concept_name, s=singular_values)

async def main():
    engine = DeepSleepEngine()
    await engine.perform_deep_sleep_cycle()

if __name__ == "__main__":
    asyncio.run(main())
