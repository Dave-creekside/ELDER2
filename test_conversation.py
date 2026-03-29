#!/usr/bin/env python3
"""Quick test: initialize ELDER2 and send a single query."""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    from streamlined_consciousness.consciousness_engine import consciousness
    from streamlined_consciousness.tool_manager import register_all_tools

    print("Registering tools...")
    register_all_tools(consciousness)

    status = consciousness.get_system_status()
    print(f"LLM: {status['llm_provider']} - {status['llm_model']}")
    print(f"Tools: {status['total_tools']} across {status['tool_categories']} categories")

    query = sys.argv[1] if len(sys.argv) > 1 else "Hello. Explore your hypergraph brain and tell me what you find."
    print(f"\n--- Sending: {query} ---\n")

    response = await consciousness.chat(query)
    print(f"\n--- Response ---\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
