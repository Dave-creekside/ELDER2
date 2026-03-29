# ELDER2

A cognitive architecture where a Claude-powered agent grows a Neo4j semantic hypergraph, and a local Gemma model learns from the agent's hidden-state traces via Riemannian-geometric LoRA consolidation.

## Architecture

```
Claude API (Opus)          Gemma-3-4b (local, 4-bit)
     |                            |
     v                            v
Neo4j Hypergraph  <----->  Shadow Tracer (PyTorch hooks)
  (concepts,                      |
   weighted edges,                v
   hyperedges)             Qdrant (trace vectors)
                                  |
                                  v
                           Deep Sleep Engine
                           (SVD + natural gradient
                            -> Hebbian LoRA updates)
                                  |
                                  v
                           Saved LoRA adapter
```

### Pipeline

1. **Consciousness Engine** — Claude API agent with LangChain tool-calling. Reads/writes a Neo4j semantic hypergraph via MCP servers. Each conversation turn queries the graph, creates concepts, and forges relationships.

2. **Shadow Tracer** — Forward hooks on Gemma's last transformer layer capture hidden-state activations (dim=2560) for each prompt+response pair. Traces are stored in Qdrant with concept anchors.

3. **Deep Sleep** — Traces are grouped by concept, decomposed via SVD to extract a metric tensor, then applied as Hebbian LoRA updates: `dB = lr * depth_scale * outer(natural_delta, A @ input) / scaling`. Layer-depth scaling (0.1 at layer 0, 1.0 at last layer) prevents early-layer disruption.

4. **Eval** — Cloze tests ("X enables ___") and graph-grounded Q&A, scored against Neo4j ground truth. Both match accuracy and perplexity are measured for base model vs tuned model.

## Current Status

### What works
- Full conversation loop: Claude explores/grows the hypergraph, Gemma captures traces, deep sleep consolidates
- 55-turn cycles complete in ~35 min (max_iterations=5, 10 tools per context)
- Perplexity consistently improves after sleep (tuned model finds graph-correct answers more plausible)
- Cloze score delta crossed positive (+0.002) — marginal but reproducible across runs
- Hausdorff dimension, R-squared, and basic graph stats available as callable metrics

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Student model | unsloth/gemma-3-4b-it | 4-bit quantized, CUDA |
| LoRA rank | 32 | Targets q/k/v/o projections |
| Deep sleep LR | 1e-5 | 100x reduction from initial 1e-3 |
| Layer scaling | 0.1 - 1.0 | Linear by depth |
| SVD rank | 32 | Adaptive 95% variance option available |
| Trace threshold | 50 | Sleep scheduler triggers at this count |
| Agent max_iterations | 5 | Prevents timeout spirals |
| Max tools per context | 10 | Caps tool selection per turn |

### Tuning results (4 runs)

| Run | Traces | LR | Cloze score delta | Cloze PPL delta | Q&A PPL delta |
|-----|--------|-----|-------------------|-----------------|---------------|
| 1 | 6 | 1e-3 | -0.067 | +13K (worse) | -366M |
| 2 | 40 | 1e-5 | -0.022 | -169K | -363M |
| 3 | 55 | 1e-5 | +0.002 | -170K | -363M |
| 4 | 53 | 1e-5 | +0.002 | -170K | -353M |

Perplexity improvement is large and consistent. Match scores are trending positive but the Hebbian update shapes probability landscapes rather than steering greedy decoding.

## Quick Start

```bash
# Prerequisites: Docker, Python 3.10+, CUDA GPU, Anthropic API key

# Start infrastructure
docker compose up -d

# Configure
cp streamlined_consciousness/.env.example streamlined_consciousness/.env
# Edit .env with your API keys

# Install
./install.sh

# Initialize Riemannian infrastructure
./venv/bin/python initialize_riemannian.py

# Interactive chat
./venv/bin/python test_conversation.py "Hello, explore your mind"

# Full cycle: conversation -> traces -> deep sleep -> eval
./venv/bin/python run_50trace_cycle.py

# Eval only (after sleep)
./venv/bin/python eval_tuning.py
```

## Project Structure

```
streamlined_consciousness/
  consciousness_engine.py  — Claude agent with tool selection
  shadow_tracer.py         — PyTorch hooks, trace capture -> Qdrant
  deep_sleep.py            — SVD + natural gradient -> LoRA updates
  student_model.py         — Gemma loading, PEFT adapter management
  tool_manager.py          — MCP tool wrappers for Neo4j/Qdrant
  consciousness_metrics.py — Hausdorff dimension, graph stats
  sleep_scheduler.py       — Background trace monitoring, auto-sleep
  config.py                — Environment-based configuration

mcp_servers/
  neo4j_hypergraph/        — Neo4j MCP server (concepts, relationships, cypher)
  qdrant_memory/           — Qdrant MCP server (vector store/search)
  sentence_transformers/   — Embedding generation

eval_tuning.py             — Base vs tuned model comparison
run_50trace_cycle.py       — Full automated cycle with status logging
run_sleep_and_eval.py      — Sleep + eval only (when traces exist)
```

## Docker Services

| Service | Container | Port | Healthcheck |
|---------|-----------|------|-------------|
| Neo4j | elder-neo4j | 7474 (HTTP), 7687 (Bolt) | wget spider |
| Qdrant | elder-qdrant | 6333 (REST), 6334 (gRPC) | bash TCP probe |

## Reset

```bash
./reset_all.sh   # Wipes Neo4j, Qdrant traces, and LoRA adapters
```
