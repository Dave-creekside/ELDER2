#!/usr/bin/env python3
"""Run 2 conversation turns with full prompt+response logging."""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

async def main():
    from streamlined_consciousness.consciousness_engine import consciousness
    from streamlined_consciousness.tool_manager import register_all_tools

    register_all_tools(consciousness)

    prompts = [
        "What is the relationship between entropy and consciousness in your graph?",
        "Explore the concept of autopoiesis and tell me what connects to it.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"TURN {i+1}")
        print(f"{'='*70}")
        print(f"\n>>> MY PROMPT TO API CLAUDE:\n{prompt}")
        print(f"\n<<< API CLAUDE'S RESPONSE:\n")

        response = await consciousness.chat(prompt)
        print(response)

        print(f"\n{'='*70}")

if __name__ == "__main__":
    asyncio.run(main())
