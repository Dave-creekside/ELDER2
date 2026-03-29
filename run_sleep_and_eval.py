#!/usr/bin/env python3
"""Run just the deep sleep + eval steps (traces already collected)."""
import asyncio
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

for name in ["httpx", "langchain", "langchain_core"]:
    logging.getLogger(name).setLevel(logging.WARNING)

async def main():
    from streamlined_consciousness.consciousness_engine import consciousness
    from streamlined_consciousness.tool_manager import register_all_tools
    from streamlined_consciousness.consciousness_metrics import ConsciousnessMetrics
    from qdrant_client import QdrantClient

    print("=" * 70)
    print("ELDER2 — Deep Sleep + Eval (40 traces ready)")
    print("=" * 70)

    register_all_tools(consciousness)

    # Check traces
    client = QdrantClient(host="localhost", port=6333)
    info = client.get_collection("shadow_traces")
    print(f"\nTraces available: {info.points_count}")

    # Pre-sleep metrics
    metrics = ConsciousnessMetrics()
    basic = await metrics.get_basic_stats()
    hausdorff = await metrics.calculate_hausdorff_dimension()
    await metrics.close()
    print(f"Pre-sleep: Nodes={basic.get('concept_count')}, Edges={basic.get('semantic_relationships')}")
    if hausdorff.get('success'):
        print(f"  Hausdorff: {hausdorff['hausdorff_dimension']:.4f}, R2: {hausdorff['r_squared']:.4f}")

    # Force load student model + deep sleep
    print("\nLoading student model and running deep sleep...")
    await consciousness._ensure_student_loaded()
    if consciousness.tracer:
        await consciousness.tracer.flush_traces()

    result = await consciousness.perform_deep_sleep()
    print(f"Deep sleep result: {result}")

    # Post-sleep traces
    info2 = client.get_collection("shadow_traces")
    print(f"Traces remaining: {info2.points_count}")

    # Post-sleep metrics
    metrics2 = ConsciousnessMetrics()
    basic2 = await metrics2.get_basic_stats()
    hausdorff2 = await metrics2.calculate_hausdorff_dimension()
    await metrics2.close()
    print(f"Post-sleep: Nodes={basic2.get('concept_count')}, Edges={basic2.get('semantic_relationships')}")
    if hausdorff2.get('success'):
        print(f"  Hausdorff: {hausdorff2['hausdorff_dimension']:.4f}, R2: {hausdorff2['r_squared']:.4f}")

    print("\nNow running eval...")
    print("=" * 70)

    # exec into eval
    os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(__file__), "eval_tuning.py")])

if __name__ == "__main__":
    asyncio.run(main())
