# Build Plan: Hybrid Riemannian Consolidation System

## Project Overview

**Goal:** Create a "School" for local LLMs. A **SOTA Teacher (API)** drives the curriculum via a **Hypergraph**, while a **Local Student (LoRA)** learns by shadowing the teacher. The learning is consolidated during sleep using Riemannian geometry, producing a portable, high-quality LoRA adapter.

**Core Loop:**
1.  **Wake:** Local Student shadows real interactions to gather context traces.
2.  **REM:** Teacher drives Student to explore distant connections in the Hypergraph.
3.  **Deep Sleep:** Traces are metabolized into LoRA weights via geometric projection.
4.  **Graduation:** The LoRA adapter is exported for standalone use.

---

## Phase 1: Infrastructure Setup

### 1.1 Neo4j Schema (The Semantic Map)
Stores the topology and the geometric curvature of concepts.

```cypher
// Node Schema
MATCH (n:Concept)
SET n.basis_vectors = [],       // Flattened U matrix (4096 x 32)
    n.singular_values = [],     // Variance strengths
    n.embedding_centroid = [],  // Current position in Local Model space
    n.last_sleep_update = 0
```

### 1.2 Qdrant Collections (Dual-Memory)

**A. Semantic Memories (The Teacher's Journal)**
*   **Use:** Standard RAG for the API Model.
*   **Vector:** 1536 (OpenAI/SOTA).
*   **Persistence:** Permanent.

**B. Shadow Traces (The Student's Synapse Buffer)**
*   **Use:** Raw material for the "Sleep" calculation.
*   **Vector:** 4096 (Local Model Hidden Dim).
*   **Persistence:** Ephemeral (Wiped after sleep).
*   **Payload:** `{"anchor_node": UUID, "state": "wake"|"rem", "input_mean": []}`

---

## Phase 2: The Trace Pipeline (Wake & REM)

### 2.1 The Shadow Tracer
A wrapper for the Local Model that captures internal movements.

```python
class ShadowTracer:
    def __init__(self, local_model, target_layers):
        self.model = local_model
        self.buffer = []

    def capture_pass(self, input_ids):
        # Run forward pass (No Gradients)
        # Hook: Capture Delta (Output - Input) at target_layers
        # Hook: Capture Input Mean (for Hebbian pairing)
        # Push to self.buffer
        pass
        
    def flush_to_qdrant(self):
        # Batch write buffer to 'shadow_traces'
        pass
```

### 2.2 Wake Cycle (Shadow Mode)
*   **Trigger:** User talks to the System.
*   **Action:** 
    1.  Teacher (API) generates the response for the user.
    2.  Student (Local) runs the *exact same prompt* in the background.
    3.  `ShadowTracer` records how the Student *would* have thought about it.

### 2.3 REM Cycle (Dreaming)
*   **Trigger:** Idle state.
*   **Teacher Action:** Queries Neo4j for unconnected concepts ($A, B$). Generates a "Lesson Prompt" bridging them.
*   **Student Action:** Runs the Lesson Prompt.
*   **Goal:** The Student forces a path through its latent space connecting $A \to B$. This path is likely "curved" (inefficient). The traces capture this curvature.

---

## Phase 3: The Deep Sleep Engine (Consolidation)

**Objective:** Mathematically process traces into LoRA weight updates.

### 3.1 Geometric Estimation (SVD)
*   **Input:** Fetch all traces for a specific Concept from `shadow_traces`.
*   **Operation:** Perform Randomized SVD on the deltas.
*   **Result:** Extract top $k=32$ components ($U$ Matrix). This maps the "local definition" of the concept.
*   **Storage:** Update `n.basis_vectors` in Neo4j.

### 3.2 Transport & Accumulation
*   **Input:** Traces from REM cycle (distant connections).
*   **Operation:**
    1.  **Pole Ladder:** Transport vectors from the "Dream" location to the nearest "Anchor" location.
    2.  **Natural Gradient:** Multiply trace $\delta$ by the inverse metric $G^{-1}$ (using Sherman-Morrison).
    3.  **Hebbian Pair:** Compute $\Delta W = \text{Natural\_Delta} \otimes \text{Input}$.
*   **Result:** A `grad_accumulator` matrix for the LoRA adapter.

### 3.3 Weight Application
*   **Operation:**
    1.  Load Local Model's LoRA adapter.
    2.  Add `grad_accumulator` * `learning_rate`.
    3.  **Save Adapter** (`adapter_model.bin`).
    4.  **Clear 'shadow_traces' Collection**.

---

## Phase 4: Integration & Mode Switching

### 4.1 Model Selector
Controls who is driving based on the maturity of the Student.

```python
class ModelSelector:
    def generate(self, prompt):
        if mode == "APPRENTICE":
            # Teacher generates, Student shadows
            return teacher.generate(prompt)
        elif mode == "JOURNEYMAN":
            # Student generates, Teacher supervises (fallback)
            return student.generate(prompt)
```

---

## Phase 5: Graduation & Deployment

**Objective:** Export the result.

### 5.1 The Merge Script
*   Load Base Model (e.g., Llama-3).
*   Load Trained LoRA Adapter.
*   Run `model.merge_and_unload()`.
*   Save as full model or GGUF.

### 5.2 Hub Upload
*   Upload the Adapter (or GGUF) to Hugging Face.
*   **End State:** The "Mind" created by your system is now a downloadable file usable in Ollama/LM Studio.
