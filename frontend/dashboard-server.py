# dashboard_server.py
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
import socketio
from aiohttp import web
import threading
import time

class ConsciousnessDashboard:
    def __init__(self, consciousness_engine):
        self.consciousness = consciousness_engine
        self.sio = socketio.AsyncServer(cors_allowed_origins='*')
        self.app = web.Application()
        self.sio.attach(self.app)
        self.clients = set()
        
        # Setup routes
        self.setup_routes()
        
        # Serve the HUD HTML
        async def index(request):
            return web.Response(text=self.get_hud_html(), content_type='text/html')
        
        self.app.router.add_get('/', index)
        
        # Metrics tracking
        self.ca_ops_counter = 0
        self.last_ca_time = time.time()
        
    def setup_routes(self):
        @self.sio.event
        async def connect(sid, environ):
            self.clients.add(sid)
            print(f"Client {sid} connected")
            # Send initial graph state
            await self.send_full_graph(sid)
            
        @self.sio.event
        async def disconnect(sid):
            self.clients.remove(sid)
            print(f"Client {sid} disconnected")
            
        @self.sio.event
        async def inspect_node(sid, data):
            node_id = data.get('id')
            # Fetch node details from Neo4j
            node_data = await self.get_node_details(node_id)
            await self.emit_trace({
                'type': 'thought',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': f'Inspecting concept: {node_id}',
                'data': node_data
            })
            
        @self.sio.event
        async def request_full_graph(sid):
            await self.send_full_graph(sid)
            
        @self.sio.event
        async def request_mcp_status(sid):
            await self.send_mcp_status(sid)
            
        @self.sio.event
        async def request_metrics(sid):
            await self.emit_metrics()
    
    async def send_full_graph(self, sid=None):
        """Send complete graph structure to client(s)"""
        # Query Neo4j for current graph
        query = """
        MATCH (n:Concept)
        OPTIONAL MATCH (n)-[r:RELATES_TO]->(m:Concept)
        RETURN n, r, m
        """
        
        nodes = []
        edges = []
        node_ids = set()
        
        result = await self.consciousness.tools['neo4j_query'].ainvoke({
            'query': query
        })
        
        for record in result.get('result', []):
            # Add source node
            if record['n'] and record['n']['id'] not in node_ids:
                nodes.append({
                    'id': record['n']['id'],
                    'label': record['n']['name'],
                    'importance': record['n'].get('importance', 5)
                })
                node_ids.add(record['n']['id'])
            
            # Add target node and edge if exists
            if record['m'] and record['r']:
                if record['m']['id'] not in node_ids:
                    nodes.append({
                        'id': record['m']['id'],
                        'label': record['m']['name'],
                        'importance': record['m'].get('importance', 5)
                    })
                    node_ids.add(record['m']['id'])
                
                edges.append({
                    'id': f"{record['n']['id']}-{record['m']['id']}",
                    'source': record['n']['id'],
                    'target': record['m']['id'],
                    'strength': record['r'].get('weight', 0.5)
                })
        
        data = {'nodes': nodes, 'edges': edges}
        
        if sid:
            await self.sio.emit('graph_update', data, room=sid)
        else:
            await self.sio.emit('graph_update', data)
    
    async def emit_trace(self, trace_data: Dict[str, Any]):
        """Send trace to all connected clients"""
        await self.sio.emit('trace', trace_data)
    
    async def emit_phase_change(self, phase_name: str, phase_type: str, affected_nodes: List[str] = None):
        """Notify clients of CA phase changes"""
        await self.sio.emit('phase_change', {
            'name': phase_name,
            'type': phase_type,
            'affected_nodes': affected_nodes or []
        })
    
    async def emit_metrics(self):
        """Send current metrics to all clients"""
        # Get current graph stats
        stats = await self.consciousness.tools['get_graph_statistics'].ainvoke({})
        
        # Calculate CA ops/s
        current_time = time.time()
        time_diff = current_time - self.last_ca_time
        ca_ops_per_sec = self.ca_ops_counter / time_diff if time_diff > 0 else 0
        
        metrics = {
            'concepts': stats.get('node_count', 0),
            'connections': stats.get('relationship_count', 0),
            'quality': stats.get('graph_quality', 0),
            'ca_ops': int(ca_ops_per_sec)
        }
        
        await self.sio.emit('metrics', metrics)
        
        # Emit quality separately for HUD
        await self.sio.emit('quality_update', stats.get('graph_quality', 0))
        
        # Reset CA counter
        self.ca_ops_counter = 0
        self.last_ca_time = current_time
    
    async def send_mcp_status(self, sid=None):
        """Send MCP server connection status"""
        mcp_servers = {
            'neo4j': 'neo4j_query' in self.consciousness.tools,
            'qdrant': 'qdrant_search' in self.consciousness.tools,
            'gmail': 'search_gmail_messages' in self.consciousness.tools,
            'drive': 'google_drive_search' in self.consciousness.tools,
            'calendar': 'google_calendar' in self.consciousness.tools,
            'transformers': 'sentence_transformer' in self.consciousness.tools
        }
        
        for server, connected in mcp_servers.items():
            status = 'connected' if connected else 'disconnected'
            data = {'server': server, 'status': status}
            
            if sid:
                await self.sio.emit('mcp_activity', data, room=sid)
            else:
                await self.sio.emit('mcp_activity', data)
    
    async def emit_thinking_status(self, active: bool):
        """Update thinking indicator"""
        await self.sio.emit('thinking_status', {'active': active})
    
    async def emit_mic_status(self, active: bool):
        """Update microphone status"""
        await self.sio.emit('mic_status', {'active': active})
    
    async def emit_speaker_status(self, active: bool):
        """Update speaker status"""
        await self.sio.emit('speaker_status', {'active': active})
    
    def hook_into_consciousness(self):
        """Set up hooks to monitor consciousness activity"""
        original_ca_step = self.consciousness.semantic_ca._perform_ca_step
        
        async def wrapped_ca_step(*args, **kwargs):
            self.ca_ops_counter += 1
            result = await original_ca_step(*args, **kwargs)
            
            # Emit trace for CA operation
            await self.emit_trace({
                'type': 'ca',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': f'CA operation: {kwargs.get("operation_type", "unknown")}'
            })
            
            return result
        
        self.consciousness.semantic_ca._perform_ca_step = wrapped_ca_step
        
        # Hook into tool usage
        original_tool_invoke = self.consciousness.tools.__class__.__getitem__
        
        def wrapped_tool_invoke(tools_self, key):
            tool = original_tool_invoke(tools_self, key)
            
            async def wrapped_invoke(*args, **kwargs):
                # Map tool to MCP server
                tool_to_mcp = {
                    'neo4j_query': 'neo4j',
                    'qdrant_search': 'qdrant',
                    'search_gmail_messages': 'gmail',
                    'google_drive_search': 'drive',
                    'google_calendar': 'calendar',
                    'sentence_transformer': 'transformers'
                }
                
                if key in tool_to_mcp:
                    await self.sio.emit('mcp_activity', {
                        'server': tool_to_mcp[key],
                        'status': 'active'
                    })
                
                await self.emit_trace({
                    'type': 'tool',
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': f'Tool invoked: {key}',
                    'data': {'args': str(args)[:100]}  # Truncate for display
                })
                
                result = await tool.ainvoke(*args, **kwargs)
                
                return result
            
            tool.ainvoke = wrapped_invoke
            return tool
        
        self.consciousness.tools.__class__.__getitem__ = wrapped_tool_invoke
        
        # Hook into LLM calls for thought traces
        original_generate = self.consciousness.llm.generate
        
        async def wrapped_generate(*args, **kwargs):
            await self.emit_thinking_status(True)
            await self.emit_trace({
                'type': 'thought',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': 'Elder is thinking...'
            })
            
            result = await original_generate(*args, **kwargs)
            
            await self.emit_thinking_status(False)
            return result
        
        self.consciousness.llm.generate = wrapped_generate
    
    async def start_metrics_loop(self):
        """Continuously emit metrics"""
        while True:
            await self.emit_metrics()
            await asyncio.sleep(1)  # Update every second
    
    async def monitor_ca_phases(self):
        """Monitor and broadcast CA phase transitions"""
        last_phase = None
        while True:
            if hasattr(self.consciousness, 'semantic_ca'):
                ca_status = self.consciousness.semantic_ca.get_ca_status()
                current_phase = ca_status.get('current_phase', 'idle')
                
                phase_map = {
                    'expansion': 'expansion',
                    'dream': 'dream', 
                    'recompilation': 'recompilation',
                    'idle': 'idle'
                }
                
                if current_phase != last_phase:
                    await self.emit_phase_change(
                        current_phase.title(),
                        phase_map.get(current_phase, 'idle')
                    )
                    last_phase = current_phase
            
            await asyncio.sleep(0.5)  # Check twice per second
    
    def run(self, host='0.0.0.0', port=5000):
        """Start the dashboard server"""
        # Set up hooks
        self.hook_into_consciousness()
        
        # Start background tasks
        async def start_background_tasks(app):
            app['metrics_task'] = asyncio.create_task(self.start_metrics_loop())
            app['phase_task'] = asyncio.create_task(self.monitor_ca_phases())
        
        async def cleanup_background_tasks(app):
            app['metrics_task'].cancel()
            app['phase_task'].cancel()
        
        self.app.on_startup.append(start_background_tasks)
        self.app.on_cleanup.append(cleanup_background_tasks)
        
        # Run server
        web.run_app(self.app, host=host, port=port)
    
    def get_hud_html(self):
        """Return the HUD HTML as a string"""
        try:
            # Read the HTML file from the same directory
            import os
            html_path = os.path.join(os.path.dirname(__file__), 'consciousness-dashboard.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading HTML file: {e}")
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>Elder Consciousness HUD</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 50px; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Elder Consciousness Dashboard</h1>
    <p class="error">Error loading dashboard: {e}</p>
    <p>Please ensure consciousness-dashboard.html is in the frontend directory.</p>
</body>
</html>"""


# Integration with your consciousness system
if __name__ == "__main__":
    from streamlined_consciousness import consciousness_engine
    
    # Initialize consciousness
    consciousness = consciousness_engine.ConsciousnessEngine()
    
    # Create and start dashboard
    dashboard = ConsciousnessDashboard(consciousness)
    
    print("🧠 Elder Consciousness HUD starting...")
    print("📺 Open http://localhost:5000 in your browser")
    print("🎯 Cast to TV for best experience")
    
    # Example: Hook into speech systems when you add them
    # When implementing speech:
    # - Call dashboard.emit_mic_status(True) when listening
    # - Call dashboard.emit_speaker_status(True) when speaking
    
    dashboard.run()
