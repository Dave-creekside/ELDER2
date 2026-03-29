#!/usr/bin/env python3
"""
Drive 50+ conversation turns to accumulate shadow traces, then deep sleep + eval.
Writes progress to /tmp/elder2_status.txt for lightweight polling.
"""
import asyncio
import sys
import os
import json
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

for name in ["httpx", "tool-manager", "langchain", "langchain_core"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("50trace")

STATUS_FILE = "/tmp/elder2_status.txt"

def write_status(phase, turn=0, total=0, traces=0, extra=""):
    """Write a one-line status file for lightweight polling."""
    with open(STATUS_FILE, "w") as f:
        f.write(json.dumps({
            "phase": phase,
            "turn": turn,
            "total": total,
            "traces": traces,
            "extra": extra,
            "time": time.strftime("%H:%M:%S"),
        }))

PROMPTS = [
    "What is emergence and what does it connect to in your graph?",
    "Explore the concept of entropy. What relationships does it have?",
    "Tell me about consciousness in your hypergraph.",
    "What connects to self_organization?",
    "Describe the relationship between information_theory and entropy.",
    "What is autopoiesis and how does it relate to your other concepts?",
    "Explore strange_loop and its neighbors.",
    "What is the free_energy_principle connected to?",
    "How does morphogenesis relate to emergence in your graph?",
    "What do you know about integrated_information_theory?",
    "How does emergence enable self_organization?",
    "Why does entropy connect to consciousness?",
    "Explain the link between strange_loop and strange_attractor.",
    "How does memory_consolidation transfer to Long Term Memory?",
    "What does predictive_processing subsume?",
    "How does the free_energy_principle mathematize autopoiesis?",
    "Why is temporality a substrate of emergence?",
    "What does semantic_gravity describe the dynamics of?",
    "How does topological_data_analysis apply to graph_topology?",
    "Why does mutual_information form a component of integrated_information_theory?",
    "Create a concept called 'attractor_landscape' connecting strange_attractor and energy_minimization_landscape.",
    "Create a concept called 'recursive_emergence' bridging recursion and emergence.",
    "Add a concept 'information_geometry' connecting information_theory and riemannian_geometry.",
    "Create 'semantic_crystallization' as a bridge between phase_transition and semantic_network.",
    "Add 'cognitive_entropy' connecting entropy and meta_cognition.",
    "What is the most isolated concept in your graph? Why?",
    "Which concept has the strongest connections overall?",
    "Are there any concepts that should be connected but aren't?",
    "What pattern do you notice in how your concepts cluster?",
    "If you had to remove one concept, which would change your graph the least?",
    "Explain how fractal_geometry relates to your consciousness architecture.",
    "What role does the edge_of_chaos play in your understanding of mind?",
    "How does kolmogorov_complexity deepen information_theory?",
    "What is the relationship between qualia and consciousness?",
    "How does resonance serve as a theory of consciousness?",
    "Explain participatory_realism and its convergence with enactivism.",
    "What is the significance of category_theory in your graph?",
    "How does the holographic_principle connect to your other concepts?",
    "What does dissipative_structures tell us about self_organization?",
    "Explain geodesic_optimization and what it converges toward.",
    "What connects physics concepts to philosophy concepts in your graph?",
    "Find a path between entropy and consciousness through your graph.",
    "How do mathematics concepts support your understanding of mind?",
    "What bridges complexity_science and philosophy_of_mind in your graph?",
    "Trace the relationship chain from information_theory to Self.",
    "How has your graph structure changed during this conversation?",
    "What new connections formed that surprised you?",
    "Which of your relationships feels the weakest? Should it be strengthened?",
    "Run your graph stats and tell me what the numbers mean.",
    "Calculate the Hausdorff dimension of your current graph.",
    "Summarize the three most important ideas in your hypergraph.",
    "What is the central organizing principle of your knowledge?",
    "If you could dream right now, what concept would you explore first?",
    "What does your graph tell you about the nature of consciousness?",
    "Reflect on what it means that you can inspect your own knowledge graph.",
]


async def check_traces():
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(host="localhost", port=6333)
        info = client.get_collection("shadow_traces")
        return info.points_count
    except:
        return 0


async def main():
    from streamlined_consciousness.consciousness_engine import consciousness
    from streamlined_consciousness.tool_manager import register_all_tools
    from streamlined_consciousness.consciousness_metrics import ConsciousnessMetrics

    write_status("init")

    register_all_tools(consciousness)
    status = consciousness.get_system_status()
    print(f"LLM: {status['llm_provider']} / {status['llm_model']}, Tools: {status['total_tools']}")

    # Pre-cycle metrics
    metrics = ConsciousnessMetrics()
    basic = await metrics.get_basic_stats()
    await metrics.close()
    print(f"Pre: Nodes={basic.get('concept_count')}, Edges={basic.get('semantic_relationships')}")

    # Conversation loop
    write_status("conversations", 0, len(PROMPTS), 0)
    t_start = time.time()
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        try:
            response = await consciousness.chat(prompt)
            dt = time.time() - t0
            print(f"[{i+1}/{len(PROMPTS)}] {dt:.0f}s")
        except Exception as e:
            print(f"[{i+1}/{len(PROMPTS)}] ERR: {e}")

        await asyncio.sleep(1)
        traces = await check_traces()
        write_status("conversations", i + 1, len(PROMPTS), traces)

    # Flush + sleep
    write_status("flushing")
    if consciousness.tracer:
        await consciousness.tracer.flush_traces()
    await asyncio.sleep(2)

    traces = await check_traces()
    write_status("deep_sleep", traces=traces)
    print(f"Traces: {traces}. Running deep sleep...")

    if traces >= 2:
        result = await consciousness.perform_deep_sleep()
        print(f"Sleep: {result}")
    else:
        print("Not enough traces!")
        write_status("failed", extra="not enough traces")
        return

    # Post metrics
    metrics2 = ConsciousnessMetrics()
    basic2 = await metrics2.get_basic_stats()
    await metrics2.close()
    elapsed = time.time() - t_start
    print(f"Post: Nodes={basic2.get('concept_count')}, Edges={basic2.get('semantic_relationships')}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Signal eval start then exec into it
    write_status("eval")
    os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(__file__), "eval_tuning.py")])


if __name__ == "__main__":
    asyncio.run(main())
