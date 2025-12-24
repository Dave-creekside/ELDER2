# Conceptual Overview: The Hybrid Teacher-Student Architecture

## The Problem
We want a local AI that evolves and learns from experience ("Self-Modifying"). However:
1.  **Cost:** Training huge models is expensive.
2.  **Quality:** Local models often hallucinate when learning on their own.
3.  **Stability:** Continual learning often breaks existing knowledge ("Catastrophic Forgetting").

## The Solution
A **Hybrid School** architecture. We use a smart, static API model to teach a small, plastic local model.

### 1. The Teacher (Cartographer)
*   **Who:** A SOTA model (GPT-4o, Claude 3.5, etc.).
*   **Role:** Holds the "True Map" of logic in the **Hypergraph** (Neo4j). It generates high-quality training scenarios ("Dreams").

### 2. The Student (Traveler)
*   **Who:** A Local Model (Llama-3, Mistral) + **LoRA Adapter**.
*   **Role:** It tries to navigate the Teacher's map. We do not update the base model; we only update the **LoRA Adapter** (The "Mind").

## The Biological Cycle

### Phase 1: Wake (Observation)
The User asks the Teacher a question. The Student listens in the background ("Shadowing"). We capture the **Trace**—the difference between the Student's input and output. This tells us: *"How did the Student interpret this reality?"*

### Phase 2: REM (Directed Dreaming)
The Teacher looks at the Hypergraph, finds two concepts that should be connected, and forces the Student to think about them. The Student generates a path in its latent space. If the path is "wobbly" (curved), it means the Student doesn't understand the connection yet.

### Phase 3: Deep Sleep (Consolidation)
The system goes offline to process the math.
1.  **Metric Estimation:** We measure the "Curvature" of the Student's knowledge. Highly curved areas are "dense facts" (don't touch!). Flat areas are "blank spaces" (safe to learn).
2.  **Riemannian Update:** We calculate a weight update that pushes the Student's "wobbly path" to become a "straight line" (an intuition).
3.  **Application:** We save these changes to the LoRA adapter.

## The Result: Portability
The final product is not the complex Hypergraph system. The final product is the **LoRA Adapter file**.
*   It contains the distilled wisdom of the Teacher.
*   It has the geometric structure of the Hypergraph.
*   It can be uploaded to Hugging Face and run on any laptop using Ollama or LM Studio.
