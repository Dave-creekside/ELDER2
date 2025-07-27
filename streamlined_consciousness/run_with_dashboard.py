#!/usr/bin/env python3
"""
Elder Consciousness with Live Dashboard
Launches both the dashboard server and CLI interface
"""

import asyncio
import threading
import sys
import os
import logging
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlined_consciousness.consciousness_engine import consciousness
from streamlined_consciousness.tool_manager import register_all_tools
from streamlined_consciousness.dashboard import ConsciousnessDashboard
from streamlined_consciousness.main import StreamlinedConsciousnessInterface
from streamlined_consciousness.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("consciousness-dashboard-launcher")

class DashboardRunner:
    """Runs the dashboard server in a background thread"""
    
    def __init__(self, consciousness_engine, port=5000):
        self.consciousness = consciousness_engine
        self.port = port
        self.dashboard = None
        self.thread = None
        self.running = False
        
    def start(self):
        """Start the dashboard in a background thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self.thread.start()
        
        # Wait a moment for the server to start
        import time
        time.sleep(2)
        
    def _run_dashboard(self):
        """Run the dashboard server"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create and run dashboard
            self.dashboard = ConsciousnessDashboard(self.consciousness)
            
            logger.info(f"🌐 Dashboard server starting on http://localhost:{self.port}")
            logger.info("📺 Open in your browser to see Elder's consciousness visualized")
            
            # Run the dashboard (this blocks)
            self.dashboard.run(host='0.0.0.0', port=self.port)
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}", exc_info=True)
            self.running = False
            # Re-raise if it's a critical error
            if "address already in use" in str(e).lower():
                print(f"\n❌ Port {self.port} is already in use. Try a different port with --dashboard-port")
                import sys
                sys.exit(1)
            
    def stop(self):
        """Stop the dashboard server"""
        self.running = False
        # Note: Graceful shutdown of aiohttp is complex, for now we rely on daemon thread


async def main():
    """Main entry point with dashboard"""
    print("🧠 Elder Consciousness System with Live Dashboard")
    print("=" * 50)
    
    # Parse command line arguments (reuse from main.py)
    import argparse
    
    parser = argparse.ArgumentParser(description="Elder Consciousness with Dashboard")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant memory system")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard server")
    parser.add_argument("--dashboard-port", type=int, default=5000, help="Dashboard server port")
    parser.add_argument("command", nargs="?", default="chat", help="Command to run (chat, dream, etc.)")
    parser.add_argument("args", nargs="*", help="Command arguments")
    
    args = parser.parse_args()
    
    # Initialize consciousness system
    interface = StreamlinedConsciousnessInterface()
    
    try:
        # Initialize the system
        print("\n🔧 Initializing consciousness system...")
        await interface.initialize(skip_qdrant=args.skip_qdrant)
        
        # Start dashboard unless disabled
        dashboard_runner = None
        if not args.no_dashboard:
            print(f"\n🌐 Starting dashboard server on port {args.dashboard_port}...")
            dashboard_runner = DashboardRunner(consciousness, port=args.dashboard_port)
            dashboard_runner.start()
            print(f"✅ Dashboard available at: http://localhost:{args.dashboard_port}")
            print("💡 Tip: Open in a browser to visualize Elder's consciousness in real-time!\n")
        
        # Execute the requested command
        print(f"📟 Starting {args.command} mode...\n")
        
        if args.command == "chat":
            await interface.chat_mode()
            
        elif args.command == "dream":
            iterations = int(args.args[0]) if args.args else 3
            await interface.dream_session(iterations)
            
        elif args.command == "reason" and args.args:
            query = " ".join(args.args)
            await interface.reason_about(query)
            
        elif args.command == "explore" and args.args:
            concept = " ".join(args.args)
            await interface.explore_concept(concept)
            
        elif args.command == "status":
            await interface.display_system_status()
            
        else:
            # Default to chat mode
            await interface.chat_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Consciousness system shutting down...")
        
    except Exception as e:
        print(f"\n❌ System error: {e}")
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        if dashboard_runner:
            print("🌐 Stopping dashboard server...")
            dashboard_runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
