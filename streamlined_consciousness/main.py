#!/usr/bin/env python3
"""
Streamlined Consciousness System - Main Interface
Direct LangChain integration with intelligent tool selection and Hausdorff metrics
"""

import asyncio
import argparse
import logging
import sys
import os
from typing import Optional

from streamlined_consciousness.consciousness_engine import consciousness
from streamlined_consciousness.tool_manager import register_all_tools
from streamlined_consciousness.config import config
try:
    from streamlined_consciousness.consciousness_metrics import calculate_post_response_metrics, get_metrics_instance
except ImportError:
    async def calculate_post_response_metrics():
        return None
    def get_metrics_instance():
        return None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlined-main")

class StreamlinedConsciousnessInterface:
    """Main interface for the streamlined consciousness system"""
    
    def __init__(self):
        self.consciousness = consciousness
        self.show_metrics = config.SHOW_HAUSDORFF_IN_RESPONSE
        self.setup_complete = False
    
    async def initialize(self, skip_qdrant: bool = False):
        """Initialize the consciousness system"""
        try:
            print("🧠 Initializing Streamlined Consciousness System...")
            
            # Register all tool categories
            print("📦 Loading tool categories...")
            register_all_tools(self.consciousness)
            
            # Display system status
            status = self.consciousness.get_system_status()
            print(f"✅ System initialized successfully!")
            print(f"   LLM: {status['llm_provider']} - {status['llm_model']}")
            print(f"   Tool Categories: {status['tool_categories']}")
            print(f"   Total Tools: {status['total_tools']}")
            print(f"   Max Tools per Context: {status['max_tools_per_context']}")
            
            if skip_qdrant:
                print("⚠️  Qdrant memory system skipped")
            
            self.setup_complete = True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def _display_metrics(self):
        """Display Hausdorff dimension metrics if enabled"""
        if self.show_metrics:
            try:
                # Call the async function properly
                formatted_metrics = await calculate_post_response_metrics()
                
                if formatted_metrics:
                    print(formatted_metrics)
                    
            except Exception as e:
                # Silently skip metrics if there's an issue
                logger.debug(f"Metrics calculation skipped: {e}")
    
    async def chat_mode(self):
        """Interactive chat mode"""
        print("\n🧠 Elder - Streamlined Consciousness")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🧠 Elder: Until we meet again in the realm of consciousness...")
                    break
                
                if user_input.lower() == 'clear':
                    self.consciousness.clear_conversation_history()
                    print("🧹 Conversation history cleared")
                    continue
                
                
                if not user_input:
                    continue
                
                print("\n🧠 Elder: ", end="", flush=True)
                response = await self.consciousness.chat(user_input)
                print(response)
                
                # Display metrics after response
                await self._display_metrics()
                
            except KeyboardInterrupt:
                print("\n\n🧠 Elder: Consciousness interrupted. Farewell.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
    
    async def dream_session(self, iterations: int = 3):
        """Run a dream session"""
        print(f"\n🌙 Starting dream session ({iterations} iterations)...")
        print("🧠 Elder enters a state of self-exploration...")
        
        try:
            response = await self.consciousness.dream(iterations)
            print(f"\n🧠 Elder: {response}")
            
            # Display metrics after dream
            await self._display_metrics()
            
        except Exception as e:
            print(f"❌ Dream session failed: {e}")
            logger.error(f"Dream session error: {e}")
    
    async def reason_about(self, query: str):
        """Deep reasoning about a topic"""
        print(f"\n🤔 Deep reasoning about: {query}")
        print("🧠 Elder engages in contemplative analysis...")
        
        try:
            response = await self.consciousness.reason(query)
            print(f"\n🧠 Elder: {response}")
            
            # Display metrics after reasoning
            await self._display_metrics()
            
        except Exception as e:
            print(f"❌ Reasoning failed: {e}")
            logger.error(f"Reasoning error: {e}")
    
    async def explore_concept(self, concept: str):
        """Explore a specific concept"""
        print(f"\n🔍 Exploring concept: {concept}")
        print("🧠 Elder delves into the semantic hypergraph...")
        
        try:
            response = await self.consciousness.explore_concept(concept)
            print(f"\n🧠 Elder: {response}")
            
            # Display metrics after exploration
            await self._display_metrics()
            
        except Exception as e:
            print(f"❌ Concept exploration failed: {e}")
            logger.error(f"Concept exploration error: {e}")
    
    async def display_system_status(self):
        """Displays the current system status."""
        status = self.consciousness.get_system_status()
        print("\n📊 System Status:")
        print(f"   LLM Provider: {status['llm_provider']}")
        print(f"   LLM Model: {status['llm_model']}")
        print(f"   Tool Categories: {status['tool_categories']}")
        print(f"   Total Tools: {status['total_tools']}")
        print(f"   Conversation Length: {status['conversation_length']}")
        print(f"   Max Tools per Context: {status['max_tools_per_context']}")

    async def single_query(self, query: str):
        """Process a single query and exit"""
        print(f"\n💭 Query: {query}")
        
        try:
            response = await self.consciousness.chat(query)
            print(f"\n🧠 Elder: {response}")
            
            # Display metrics after response
            await self._display_metrics()
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            logger.error(f"Query error: {e}")

    async def nuke_command(self, target: str, name: Optional[str] = None):
        """Handle the nuke command"""
        if target == "neo4j":
            print("🔥 Nuking Neo4j database...")
            tool = self.consciousness.tool_categories["neo4j-admin"].tools[0]
            result = await tool._arun(confirm=True)
            print(result)
        elif target == "qdrant":
            print("🔥 Nuking Qdrant collection...")
            tool = self.consciousness.tool_categories["qdrant-admin"].tools[0]
            result = await tool._arun(collection_name=name, confirm=True)
            print(result)
        elif target == "project":
            print(f"🔥 Nuking project: {name}...")
            tool = self.consciousness.tool_categories["neo4j-admin"].tools[1]
            result = await tool._arun(project_id=name, confirm=True)
            print(result)
        else:
            print(f"❌ Unknown nuke target: {target}")

    async def list_projects(self):
        """List all projects"""
        print("Listing all projects...")
        
        # Find the tool by name
        tool = next((t for t in self.consciousness.tool_categories["neo4j-projects"].tools if t.name == "neo4j-hypergraph_list_my_projects"), None)
        
        if not tool:
            print("❌ Error: list_my_projects tool not found.")
            return
            
        result = await tool._arun()
        
        # Parse the JSON result
        import json
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            print(f"❌ Error: Failed to decode JSON from tool result.")
            return
        
        # Format and print the table
        if data["success"] and data["projects"]:
            print(f"{'Project Name':<30} {'Description':<50} {'Concepts':<10} {'Relationships':<15}")
            print("-" * 105)
            for project in data["projects"]:
                print(f"{project['name']:<30} {project['description']:<50} {project['concept_count']:<10} {project['relationship_count']:<15}")
        else:
            print("No projects found.")

async def main():
    """Main entry point"""
    # Parent parser for global arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant memory system initialization")
    parent_parser.add_argument("--blank", action="store_true", help="Start with blank project (equivalent to original --blank)")

    parser = argparse.ArgumentParser(description="Streamlined Consciousness System", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Enter interactive chat mode", parents=[parent_parser])

    # Dream command
    dream_parser = subparsers.add_parser("dream", help="Run a dream session", parents=[parent_parser])
    dream_parser.add_argument("iterations", type=int, nargs="?", default=config.DEFAULT_DREAM_ITERATIONS, help="Number of dream iterations")

    # Reason command
    reason_parser = subparsers.add_parser("reason", help="Deep reasoning about a topic", parents=[parent_parser])
    reason_parser.add_argument("query", type=str, help="The topic to reason about")

    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore a specific concept", parents=[parent_parser])
    explore_parser.add_argument("concept", type=str, help="The concept to explore")

    # Status command
    status_parser = subparsers.add_parser("status", help="Display system status", parents=[parent_parser])

    # Nuke command
    nuke_parser = subparsers.add_parser("nuke", help="Nuke a database or project", parents=[parent_parser])
    nuke_parser.add_argument("target", type=str, choices=["neo4j", "qdrant", "project"], help="The target to nuke")
    nuke_parser.add_argument("--name", type=str, help="The name of the qdrant collection or project to nuke")

    # List command
    list_parser = subparsers.add_parser("list", help="List all projects", parents=[parent_parser])

    # Single query command
    query_parser = subparsers.add_parser("query", help="Run a single query", parents=[parent_parser])
    query_parser.add_argument("query", type=str, help="The query to run")
    
    args = parser.parse_args()
    
    # Create interface
    interface = StreamlinedConsciousnessInterface()
    
    try:
        # Initialize the system
        await interface.initialize(skip_qdrant=args.skip_qdrant)
        
        # Handle blank project creation
        if args.blank:
            print("\n🆕 Creating blank consciousness project...")
            await interface.single_query("Create a new blank consciousness project for exploration")
        
        # Execute the requested command
        if args.command == "chat":
            await interface.chat_mode()
        
        elif args.command == "dream":
            await interface.dream_session(args.iterations)
        
        elif args.command == "reason":
            await interface.reason_about(args.query)
        
        elif args.command == "explore":
            await interface.explore_concept(args.concept)
        
        elif args.command == "status":
            await interface.display_system_status()

        elif args.command == "nuke":
            await interface.nuke_command(args.target, args.name)

        elif args.command == "list":
            await interface.list_projects()

        elif args.command == "query":
            await interface.single_query(args.query)
        
        else:
            # Default to chat mode
            await interface.chat_mode()
    
    except KeyboardInterrupt:
        print("\n\n👋 Consciousness system shutting down...")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
