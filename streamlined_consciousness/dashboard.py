#!/usr/bin/env python3
"""
Elder Consciousness Dashboard Server
Real-time visualization of consciousness evolution
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import socketio
from aiohttp import web
import threading
import time
import os
import logging

logger = logging.getLogger("consciousness-dashboard")

class ConsciousnessDashboard:
    def __init__(self, consciousness_engine):
        self.consciousness = consciousness_engine
        self.sio = socketio.AsyncServer(cors_allowed_origins='*')
        self.app = web.Application()
        self.sio.attach(self.app)
        self.clients = set()
        
        # Setup routes
        self.setup_routes()
        
        # Serve the unified dashboard
        async def index(request):
            return web.Response(text=self.get_unified_dashboard_html(), content_type='text/html')
        
        # Also serve individual pages for backward compatibility
        async def dashboard_legacy(request):
            return web.Response(text=self.get_hud_html(), content_type='text/html')
        
        async def galaxy(request):
            return web.Response(text=self.get_galaxy_html(), content_type='text/html')
        
        async def health(request):
            return web.Response(text=self.get_health_html(), content_type='text/html')
        
        self.app.router.add_get('/', index)
        self.app.router.add_get('/dashboard', dashboard_legacy)
        self.app.router.add_get('/galaxy', galaxy)
        self.app.router.add_get('/health', health)
        
        # Serve individual HTML files for the unified dashboard to load
        async def serve_html(request):
            filename = request.match_info.get('filename')
            # Map filenames to the correct method names
            method_map = {
                'consciousness-dashboard.html': 'get_hud_html',
                'consciousness-galaxy.html': 'get_galaxy_html',
                'health.html': 'get_health_html'
            }
            
            if filename in method_map:
                method_name = method_map[filename]
                html_content = getattr(self, method_name)()
                return web.Response(text=html_content, content_type='text/html')
            return web.Response(status=404)
        
        self.app.router.add_get('/{filename:.+\.html}', serve_html)
        
        # Serve static CSS and JS files
        async def serve_static(request):
            filename = request.match_info.get('filename')
            static_files = ['elder-dashboard.css', 'elder-dashboard.js']
            
            if filename in static_files:
                file_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    content_type = 'text/css' if filename.endswith('.css') else 'application/javascript'
                    return web.Response(text=content, content_type=content_type)
            
            return web.Response(status=404)
        
        self.app.router.add_get('/{filename:.+\.(css|js)}', serve_static)
        
        # Metrics tracking
        self.ca_ops_counter = 0
        self.last_ca_time = time.time()
        
        # Neo4j connection
        self.neo4j_driver = None
        
    def setup_routes(self):
        @self.sio.event
        async def connect(sid, environ):
            self.clients.add(sid)
            logger.info(f"Dashboard client {sid} connected")
            # Send initial graph state
            await self.send_full_graph(sid)
            await self.send_mcp_status(sid)
            
        @self.sio.event
        async def disconnect(sid):
            self.clients.remove(sid)
            logger.info(f"Dashboard client {sid} disconnected")
            
        @self.sio.event
        async def inspect_node(sid, data):
            node_id = data.get('id')
            # Fetch node details from Neo4j
            await self.emit_trace({
                'type': 'thought',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'message': f'Inspecting concept: {node_id}',
                'data': {'node_id': node_id}
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
            
        @self.sio.event
        async def chat_message(sid, data):
            """Handle incoming chat messages from the terminal"""
            try:
                message = data.get('message', '').strip()
                if not message:
                    await self.sio.emit('chat_error', {
                        'error': 'Empty message'
                    }, room=sid)
                    return
                
                logger.info(f"Chat message from {sid}: {message}")
                
                # Process the message through Elder
                response = await self.consciousness.chat(message)
                
                # Send response back to client
                await self.sio.emit('chat_response', {
                    'message': response,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error processing chat message: {e}")
                await self.sio.emit('chat_error', {
                    'error': str(e)
                }, room=sid)
    
    async def send_full_graph(self, sid=None):
        """Send complete graph structure to client(s)"""
        try:
            # Find the Neo4j query tool - look for the correct tool name
            neo4j_tool = None
            for category in self.consciousness.tool_categories.values():
                for tool in category.tools:
                    # The actual tool name is neo4j-hypergraph_query_cypher
                    if tool.name == 'neo4j-hypergraph_query_cypher' or 'query_cypher' in tool.name:
                        neo4j_tool = tool
                        logger.info(f"Found Neo4j query tool: {tool.name}")
                        break
                if neo4j_tool:
                    break
            
            if not neo4j_tool:
                logger.warning("Neo4j query tool not found")
                return
            
            # Query Neo4j for current graph - separate queries to avoid issues
            # Query 1: Get all concepts with connection count for importance
            concepts_query = """
            MATCH (n:Concept)
            OPTIONAL MATCH (n)-[r:SEMANTIC]-(connected:Concept)
            WITH n, count(DISTINCT connected) as connection_count
            RETURN collect(DISTINCT {node: n, connections: connection_count}) as concepts
            """
            
            # Query 2: Get all relationships (not just SEMANTIC)
            relationships_query = """
            MATCH (n1:Concept)-[r]-(n2:Concept)
            RETURN collect(DISTINCT {n: n1, r: r, m: n2}) as relationships
            """
            
            # Query 3: Get hyperedges with members
            hyperedges_query = """
            MATCH (he:Hyperedge)
            OPTIONAL MATCH (member:Concept)-[:MEMBER_OF]->(he)
            WITH he, collect(DISTINCT member.name) as member_names
            RETURN collect({hyperedge: he, members: member_names}) as hyperedges
            """
            
            nodes = []
            edges = []
            node_ids = set()
            hyperedges = []
            
            # Run the three queries separately
            # Query 1: Get concepts with connection count
            concepts_result = await self._query_neo4j_directly(concepts_query)
            if concepts_result and concepts_result.get('records'):
                record = concepts_result['records'][0]
                concepts = record.get('concepts', [])
                for concept_data in concepts:
                    if isinstance(concept_data, dict) and 'node' in concept_data:
                        node = concept_data['node']
                        connections = concept_data.get('connections', 0)
                        node_id = node.get('name', str(node))
                        if node_id not in node_ids:
                            # Use connection count as importance (scale it for better visualization)
                            # Core nodes get minimum importance of 5
                            importance = max(connections * 0.5 + 3, 5) if connections > 0 else 3
                            if node_id in ['tools', 'long_term_memory', 'working_memory']:
                                importance = max(importance, 8)  # Core nodes are always prominent
                            
                            nodes.append({
                                'id': node_id,
                                'label': node_id,
                                'importance': importance
                            })
                            node_ids.add(node_id)
            
            # Query 2: Get relationships (if any)
            relationships_result = await self._query_neo4j_directly(relationships_query)
            if relationships_result and relationships_result.get('records') and len(relationships_result['records']) > 0:
                record = relationships_result['records'][0]
                relationships = record.get('relationships', [])
                processed_edges = set()
                for rel_data in relationships:
                    if rel_data and rel_data.get('n') and rel_data.get('m') and rel_data.get('r'):
                        source_id = rel_data['n'].get('name', str(rel_data['n']))
                        target_id = rel_data['m'].get('name', str(rel_data['m']))
                        
                        # Avoid duplicate edges (since relationships are bidirectional)
                        edge_key = tuple(sorted([source_id, target_id]))
                        if edge_key not in processed_edges:
                            processed_edges.add(edge_key)
                            edges.append({
                                'id': f"{source_id}-{target_id}",
                                'source': source_id,
                                'target': target_id,
                                'strength': rel_data['r'].get('weight', rel_data['r'].get('semantic_weight', 0.5))
                            })
            
            # Query 3: Get hyperedges
            hyperedges_result = await self._query_neo4j_directly(hyperedges_query)
            if hyperedges_result and hyperedges_result.get('records') and len(hyperedges_result['records']) > 0:
                record = hyperedges_result['records'][0]
                hyperedges_data = record.get('hyperedges', [])
                for he_data in hyperedges_data:
                    if he_data and he_data.get('hyperedge') and he_data.get('members'):
                        hyperedge = he_data['hyperedge']
                        members = he_data['members']
                        if len(members) > 1:  # Only include hyperedges with multiple members
                            hyperedges.append({
                                'id': hyperedge.get('id', str(hyperedge)),
                                'label': hyperedge.get('label', 'hyperedge'),
                                'members': members,
                                'collective_weight': hyperedge.get('collective_weight', 1.0),
                                'member_count': hyperedge.get('member_count', len(members))
                            })
            
            data = {'nodes': nodes, 'edges': edges, 'hyperedges': hyperedges}
            
            logger.info(f"Sending graph data: {len(nodes)} nodes, {len(edges)} edges, {len(hyperedges)} hyperedges")
            
            if sid:
                await self.sio.emit('graph_update', data, room=sid)
            else:
                await self.sio.emit('graph_update', data)
                
        except Exception as e:
            logger.error(f"Error sending graph update: {e}")
    
    async def emit_trace(self, trace_data: Dict[str, Any]):
        """Send trace to all connected clients"""
        await self.sio.emit('trace', trace_data)
    
    async def emit_edge_event(self, event_type: str, source: str, target: str, 
                            weight: float = None, metadata: Dict[str, Any] = None):
        """Emit edge manipulation events for real-time visualization"""
        event_data = {
            'type': event_type,  # 'created', 'accessed', 'strengthened', 'weakened', 'pruned'
            'source': source,
            'target': target,
            'timestamp': datetime.now().isoformat(),
            'weight': weight,
            'metadata': metadata or {}
        }
        await self.sio.emit('edge_event', event_data)
    
    async def emit_phase_change(self, phase_name: str, phase_type: str, affected_nodes: List[str] = None):
        """Notify clients of CA phase changes"""
        await self.sio.emit('phase_change', {
            'name': phase_name,
            'type': phase_type,
            'affected_nodes': affected_nodes or []
        })
    
    async def emit_metrics(self):
        """Send current metrics to all clients"""
        try:
            # Find the stats tool - look for the correct tool name
            stats_tool = None
            for category in self.consciousness.tool_categories.values():
                for tool in category.tools:
                    # The actual tool name is neo4j-hypergraph_get_graph_stats
                    if tool.name == 'neo4j-hypergraph_get_graph_stats' or 'get_graph_stats' in tool.name:
                        stats_tool = tool
                        # Don't log every time - this runs every second
                        break
                if stats_tool:
                    break
            
            if stats_tool:
                # Get current graph stats
                stats_result = await asyncio.to_thread(stats_tool._run)
                if isinstance(stats_result, str):
                    stats = json.loads(stats_result)
                else:
                    stats = stats_result
            else:
                stats = {}
            
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
            
        except Exception as e:
            logger.error(f"Error emitting metrics: {e}")
    
    async def send_mcp_status(self, sid=None):
        """Send MCP server connection status"""
        # Check which tools are available based on tool categories
        mcp_servers = {
            'neo4j': False,
            'qdrant': False,
            'gmail': False,
            'drive': False,
            'calendar': False,
            'transformers': False
        }
        
        # Check tool availability
        for category in self.consciousness.tool_categories.values():
            for tool in category.tools:
                if 'neo4j' in tool.name:
                    mcp_servers['neo4j'] = True
                elif 'qdrant' in tool.name:
                    mcp_servers['qdrant'] = True
                elif 'gmail' in tool.name:
                    mcp_servers['gmail'] = True
                elif 'drive' in tool.name:
                    mcp_servers['drive'] = True
                elif 'calendar' in tool.name:
                    mcp_servers['calendar'] = True
                elif 'sentence_transformer' in tool.name:
                    mcp_servers['transformers'] = True
        
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
        # Store reference to dashboard
        self.consciousness.dashboard = self
        
        # Hook into semantic CA if available
        if hasattr(self.consciousness, 'semantic_ca') and self.consciousness.semantic_ca:
            try:
                # Hook into create_semantic_connection
                if hasattr(self.consciousness.semantic_ca, '_create_semantic_connection'):
                    original_create = self.consciousness.semantic_ca._create_semantic_connection
                    
                    async def wrapped_create(*args, **kwargs):
                        self.ca_ops_counter += 1
                        
                        # Extract connection details from args
                        concept1 = args[0] if len(args) > 0 else kwargs.get('concept1', 'unknown')
                        concept2 = args[1] if len(args) > 1 else kwargs.get('concept2', 'unknown')
                        semantic_weight = args[2] if len(args) > 2 else kwargs.get('semantic_weight', 0.5)
                        
                        result = await original_create(*args, **kwargs)
                        
                        # Emit trace for CA operation
                        await self.emit_trace({
                            'type': 'ca',
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'message': f'CA: Creating semantic connection between {concept1} and {concept2}'
                        })
                        
                        # Emit edge event for visualization
                        if result:  # Only emit if creation was successful
                            await self.emit_edge_event(
                                'created',
                                concept1,
                                concept2,
                                semantic_weight,
                                {'created_by': 'semantic_ca', 'phase': 'unknown'}
                            )
                        
                        return result
                    
                    self.consciousness.semantic_ca._create_semantic_connection = wrapped_create
                
                # Hook into prune_weak_connections
                if hasattr(self.consciousness.semantic_ca, '_prune_weak_connections'):
                    original_prune = self.consciousness.semantic_ca._prune_weak_connections
                    
                    async def wrapped_prune(*args, **kwargs):
                        self.ca_ops_counter += 1
                        result = await original_prune(*args, **kwargs)
                        
                        # Emit trace for CA operation
                        await self.emit_trace({
                            'type': 'ca',
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'message': 'CA: Pruning weak connections'
                        })
                        
                        return result
                    
                    self.consciousness.semantic_ca._prune_weak_connections = wrapped_prune
                    
            except Exception as e:
                logger.warning(f"Failed to hook into semantic CA: {e}")
        
        # Hook into chat for thinking status
        original_chat = self.consciousness.chat
        
        async def wrapped_chat(user_input: str):
            await self.emit_thinking_status(True)
            result = await original_chat(user_input)
            await self.emit_thinking_status(False)
            
            # Emit graph update after response
            await self.send_full_graph()
            
            return result
        
        self.consciousness.chat = wrapped_chat
        
        # Hook into Neo4j relationship creation and deletion tools
        self._hook_into_neo4j_tools()
        
        # Hook into dream sessions
        if hasattr(self.consciousness, 'dream'):
            original_dream = self.consciousness.dream
            
            async def wrapped_dream(iterations: int = 3):
                await self.emit_phase_change("Dream", "dream")
                result = await original_dream(iterations)
                await self.emit_phase_change("Idle", "idle")
                
                # Update graph after dream
                await self.send_full_graph()
                
                return result
            
            self.consciousness.dream = wrapped_dream
    
    async def start_metrics_loop(self):
        """Continuously emit metrics"""
        while True:
            await self.emit_metrics()
            await asyncio.sleep(5)  # Update every 5 seconds instead of every second
    
    async def monitor_ca_phases(self):
        """Monitor and broadcast CA phase transitions"""
        last_phase = None
        while True:
            if hasattr(self.consciousness, 'semantic_ca') and self.consciousness.semantic_ca:
                ca_status = self.consciousness.semantic_ca.get_ca_status()
                current_phase = ca_status.get('current_phase', 'idle')
                
                phase_map = {
                    'PRE_DREAM': 'expansion',
                    'DREAM': 'dream', 
                    'POST_DREAM': 'recompilation',
                    'IDLE': 'idle'
                }
                
                if current_phase != last_phase:
                    await self.emit_phase_change(
                        current_phase.replace('_', ' ').title(),
                        phase_map.get(current_phase, 'idle')
                    )
                    last_phase = current_phase
            
            await asyncio.sleep(0.5)  # Check twice per second
    
    def _hook_into_neo4j_tools(self):
        """Hook into Neo4j tools to emit edge events"""
        try:
            # Find neo4j tools to hook
            for category in self.consciousness.tool_categories.values():
                for tool in category.tools:
                    # Hook into relationship creation
                    if 'create_relationship' in tool.name:
                        original_run = tool._run
                        
                        def wrapped_run(*args, **kwargs):
                            result = original_run(*args, **kwargs)
                            
                            # Parse the result to get relationship details
                            try:
                                if isinstance(result, str):
                                    result_data = json.loads(result)
                                else:
                                    result_data = result
                                    
                                if result_data.get('success'):
                                    # Extract relationship details
                                    from_concept = kwargs.get('from_concept', 'unknown')
                                    to_concept = kwargs.get('to_concept', 'unknown')
                                    weight = result_data.get('semantic_weight', 0.5)
                                    
                                    # Emit edge event asynchronously
                                    asyncio.create_task(self.emit_edge_event(
                                        'created',
                                        from_concept,
                                        to_concept,
                                        weight,
                                        {'created_by': 'elder', 'type': kwargs.get('relationship_type', 'RELATED')}
                                    ))
                            except Exception as e:
                                logger.warning(f"Failed to emit edge event: {e}")
                            
                            return result
                        
                        tool._run = wrapped_run
                        logger.info(f"Hooked into Neo4j tool: {tool.name}")
                    
                    # Hook into node deletion
                    elif 'delete_concept_node' in tool.name:
                        original_run = tool._run
                        
                        def wrapped_delete_run(*args, **kwargs):
                            result = original_run(*args, **kwargs)
                            
                            # Parse the result to check if deletion was successful
                            try:
                                if isinstance(result, str):
                                    result_data = json.loads(result)
                                else:
                                    result_data = result
                                    
                                if result_data.get('success'):
                                    # Emit trace that node was deleted
                                    node_name = kwargs.get('name', 'unknown')
                                    asyncio.create_task(self.emit_trace({
                                        'type': 'node_deleted',
                                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                                        'message': f'Deleted concept node: {node_name}'
                                    }))
                                    
                                    # Send full graph update after deletion
                                    asyncio.create_task(self.send_full_graph())
                                    
                            except Exception as e:
                                logger.warning(f"Failed to emit deletion event: {e}")
                            
                            return result
                        
                        tool._run = wrapped_delete_run
                        logger.info(f"Hooked into Neo4j deletion tool: {tool.name}")
                        
        except Exception as e:
            logger.warning(f"Failed to hook into Neo4j tools: {e}")
    
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
        
        async def cleanup_connections(app):
            # Close Neo4j connection
            if self.neo4j_driver:
                self.neo4j_driver.close()
        
        async def init_neo4j_driver(app):
            """Initialize Neo4j driver at startup"""
            try:
                from neo4j import GraphDatabase
                from streamlined_consciousness.config import config
                
                self.neo4j_driver = GraphDatabase.driver(
                    config.NEO4J_URI, 
                    auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
                )
                logger.info("Neo4j driver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                self.neo4j_driver = None
        
        self.app.on_startup.append(init_neo4j_driver)
        self.app.on_startup.append(start_background_tasks)
        self.app.on_cleanup.append(cleanup_background_tasks)
        self.app.on_cleanup.append(cleanup_connections)
        
        # Run server with handle_signals=False for thread compatibility
        web.run_app(self.app, host=host, port=port, handle_signals=False)
    
    def get_hud_html(self):
        """Return the HUD HTML as a string"""
        try:
            # Look for HTML file in multiple locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'frontend', 'consciousness-dashboard.html'),
                os.path.join(os.path.dirname(__file__), 'consciousness-dashboard.html'),
                'frontend/consciousness-dashboard.html'
            ]
            
            for html_path in possible_paths:
                if os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            raise FileNotFoundError("consciousness-dashboard.html not found in any expected location")
            
        except Exception as e:
            logger.error(f"Error reading HTML file: {e}")
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
    
    def get_galaxy_html(self):
        """Return the Galaxy 3D visualization HTML as a string"""
        try:
            # Look for HTML file in multiple locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'frontend', 'consciousness-galaxy.html'),
                os.path.join(os.path.dirname(__file__), 'consciousness-galaxy.html'),
                'frontend/consciousness-galaxy.html'
            ]
            
            for html_path in possible_paths:
                if os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            raise FileNotFoundError("consciousness-galaxy.html not found in any expected location")
            
        except Exception as e:
            logger.error(f"Error reading Galaxy HTML file: {e}")
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>Elder Neural Galaxy</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 50px; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Elder Neural Galaxy</h1>
    <p class="error">Error loading 3D visualization: {e}</p>
    <p>Please ensure consciousness-galaxy.html is in the frontend directory.</p>
</body>
</html>"""
    
    def get_health_html(self):
        """Return the Health monitoring HTML as a string"""
        try:
            # Look for HTML file in multiple locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'frontend', 'health.html'),
                os.path.join(os.path.dirname(__file__), 'health.html'),
                'frontend/health.html'
            ]
            
            for html_path in possible_paths:
                if os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            raise FileNotFoundError("health.html not found in any expected location")
            
        except Exception as e:
            logger.error(f"Error reading Health HTML file: {e}")
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>ELDER System Health</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 50px; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>ELDER System Health</h1>
    <p class="error">Error loading health monitoring: {e}</p>
    <p>Please ensure health.html is in the frontend directory.</p>
</body>
</html>"""
    
    def get_unified_dashboard_html(self):
        """Return the unified dashboard HTML as a string"""
        try:
            # Look for HTML file in multiple locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'frontend', 'elder-dashboard.html'),
                os.path.join(os.path.dirname(__file__), 'elder-dashboard.html'),
                'frontend/elder-dashboard.html'
            ]
            
            for html_path in possible_paths:
                if os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            raise FileNotFoundError("elder-dashboard.html not found in any expected location")
            
        except Exception as e:
            logger.error(f"Error reading unified dashboard HTML file: {e}")
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>ELDER Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 50px; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>ELDER Dashboard</h1>
    <p class="error">Error loading dashboard: {e}</p>
    <p>Please ensure elder-dashboard.html is in the frontend directory.</p>
</body>
</html>"""
    
    async def _query_neo4j_directly(self, query: str, parameters: dict = None) -> dict:
        """Query Neo4j directly without MCP filtering"""
        try:
            # Check if driver exists (should be created at startup)
            if not self.neo4j_driver:
                logger.error("Neo4j driver not initialized - this should not happen!")
                return None
            
            # Run synchronous query in thread pool to avoid blocking
            def run_query():
                with self.neo4j_driver.session() as session:
                    result = session.run(query, parameters or {})
                    records = []
                    for record in result:
                        # Convert Neo4j record to dict
                        record_dict = {}
                        for key in record.keys():
                            value = record[key]
                            # Convert Neo4j nodes/relationships to dicts
                            if hasattr(value, 'items'):
                                record_dict[key] = dict(value.items())
                            else:
                                record_dict[key] = value
                        records.append(record_dict)
                    
                    return {
                        "success": True,
                        "records": records,
                        "count": len(records)
                    }
            
            # Execute in thread pool to avoid blocking async event loop
            return await asyncio.to_thread(run_query)
                
        except Exception as e:
            logger.error(f"Direct Neo4j query failed: {e}")
            return None
