#!/usr/bin/env python3
"""
Full ELDER2 cycle: conversation -> traces -> metrics -> deep sleep -> metrics.
Drives multiple conversation turns to accumulate shadow traces, then consolidates.
"""
import asyncio
import sys
import os
import json
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full-cycle")

CONVERSATION_TURNS = [
    "Explore the concept of 'emergence' in your hypergraph. What connects to it and why?",
    "Create a new concept called 'dissipative_adaptation' that bridges dissipative_structures and self_organization. Explain why this connection matters.",
    "How does your strange_loop concept relate to autopoiesis? Strengthen that connection if it exists, or create it.",
    "Reflect on the relationship between entropy and consciousness in your graph. What's missing?",
    "Dream briefly — explore one unexpected connection between two distant concepts in your mind.",
]

async def check_qdrant_traces():
    """Check how many shadow traces are in Qdrant."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections().collections
        for c in collections:
            info = client.get_collection(c.name)
            print(f"  Qdrant '{c.name}': {info.points_count} points, dim={info.config.params.vectors.size}")
        shadow = next((c for c in collections if c.name == "shadow_traces"), None)
        if shadow:
            info = client.get_collection("shadow_traces")
            return info.points_count
        return 0
    except Exception as e:
        print(f"  Qdrant check failed: {e}")
        return 0

async def run_metrics():
    """Run Hausdorff dimension and basic stats."""
    from streamlined_consciousness.consciousness_metrics import ConsciousnessMetrics
    metrics = ConsciousnessMetrics()
    basic = await metrics.get_basic_stats()
    hausdorff = await metrics.calculate_hausdorff_dimension()
    await metrics.close()
    return basic, hausdorff

async def main():
    from streamlined_consciousness.consciousness_engine import consciousness
    from streamlined_consciousness.tool_manager import register_all_tools

    print("=" * 60)
    print("ELDER2 Full Cycle Test")
    print("=" * 60)

    # Init
    print("\n[1] Initializing system...")
    register_all_tools(consciousness)
    status = consciousness.get_system_status()
    print(f"    LLM: {status['llm_provider']} / {status['llm_model']}")
    print(f"    Tools: {status['total_tools']}")

    # Pre-cycle metrics
    print("\n[2] Pre-cycle metrics...")
    basic, hausdorff = await run_metrics()
    print(f"    Nodes: {basic.get('concept_count')}, Edges: {basic.get('semantic_relationships')}")
    if hausdorff.get('success'):
        print(f"    Hausdorff dim: {hausdorff['hausdorff_dimension']:.4f}, R²: {hausdorff['r_squared']:.4f}")
    else:
        print(f"    Hausdorff: {hausdorff.get('message', hausdorff.get('error'))}")

    # Pre-cycle traces
    print("\n[3] Pre-cycle Qdrant state...")
    pre_traces = await check_qdrant_traces()

    # Conversation turns
    print(f"\n[4] Running {len(CONVERSATION_TURNS)} conversation turns...")
    for i, turn in enumerate(CONVERSATION_TURNS):
        print(f"\n--- Turn {i+1}/{len(CONVERSATION_TURNS)} ---")
        print(f"  Q: {turn[:80]}...")
        t0 = time.time()
        response = await consciousness.chat(turn)
        dt = time.time() - t0
        # Print first 300 chars of response
        print(f"  A: {response[:300]}...")
        print(f"  ({dt:.1f}s)")

        # Give background trace a moment to flush
        await asyncio.sleep(2)

    # Post-conversation traces
    print(f"\n[5] Post-conversation Qdrant state...")
    # Force flush any buffered traces
    if consciousness.tracer:
        await consciousness.tracer.flush_traces()
    await asyncio.sleep(1)
    post_traces = await check_qdrant_traces()
    print(f"    Traces: {pre_traces} -> {post_traces} (gained {post_traces - pre_traces})")

    # Deep sleep consolidation
    if post_traces > 0:
        print(f"\n[6] Triggering Deep Sleep consolidation...")
        result = await consciousness.perform_deep_sleep()
        print(f"    Result: {result}")
    else:
        print(f"\n[6] No traces to consolidate — student model may not have loaded yet.")
        print(f"    Student loaded: {consciousness.student_model is not None and consciousness.student_model.model is not None}")

    # Post-cycle metrics
    print(f"\n[7] Post-cycle metrics...")
    basic2, hausdorff2 = await run_metrics()
    print(f"    Nodes: {basic2.get('concept_count')}, Edges: {basic2.get('semantic_relationships')}")
    if hausdorff2.get('success'):
        print(f"    Hausdorff dim: {hausdorff2['hausdorff_dimension']:.4f}, R²: {hausdorff2['r_squared']:.4f}")
        if hausdorff.get('success'):
            delta = hausdorff2['hausdorff_dimension'] - hausdorff['hausdorff_dimension']
            print(f"    Delta H-dim: {delta:+.4f}")
    else:
        print(f"    Hausdorff: {hausdorff2.get('message', hausdorff2.get('error'))}")

    # Final trace count
    final_traces = await check_qdrant_traces()
    print(f"\n[8] Final Qdrant state...")
    print(f"    Remaining traces: {final_traces} (should be 0 after deep sleep)")

    print("\n" + "=" * 60)
    print("Cycle complete.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
