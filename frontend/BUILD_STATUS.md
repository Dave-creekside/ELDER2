# ELDER2 Frontend Build Status

## Stack
- **SvelteKit 2** in SPA mode (adapter-static, no SSR)
- **Svelte 5**, TypeScript, Vite 6
- **D3.js v7** — radial graph layout, force simulation for galaxy
- **Three.js v0.170** — 3D rendering with bloom post-processing
- **Socket.IO client** — connects to Python backend on port 5000
- **Node 22** required (via nvm)

## Architecture

```
src/
  lib/
    config.ts              — SERVER_URL, CORE_NODES, GRAPH params, SLEEP threshold
    stores/
      socket.ts            — singleton Socket.IO connection, connection status, debug logging (onAny)
      graph.ts             — nodes/edges/hyperedges from backend, derived metrics
      chat.ts              — messages, thinking state, send/receive
      health.ts            — health stats, deep sleep induction
    components/
      ChatTerminal.svelte  — chat UI with timestamps, thinking dots, auto-resize
      MetricCard.svelte    — reusable metric display card
    utils/
      colors.ts            — shared palette, heat interpolation, edge opacity
  routes/
    +layout.svelte         — sidebar nav, connection status, node/edge counts
    +layout.ts             — SPA mode (ssr=false)
    /                      — Chat (home page)
    /graph                 — Radial concentric ring layout (D3)
    /galaxy                — 3D force-directed with bloom (D3 + Three.js)
    /heatmap               — Canvas heat grid
    /matrix                — BFS distance matrix
    /fractal               — Scale invariance multi-radius
    /health                — System health dashboard
    /student               — Student model status
```

## Backend Data Format (dashboard.py on port 5000)

Socket events the frontend consumes:

| Event | Payload shape |
|-------|--------------|
| `graph_update` | `{nodes: [{id, label, importance}], edges: [{id, source, target, strength}], hyperedges: [{id, label, members[], collective_weight}]}` |
| `chat_response` | `{message: string, timestamp: string}` |
| `chat_error` | `{error: string}` |
| `thinking_status` | `{active: boolean}` |
| `health_stats` | `{graph: {node_count, edge_count, hyperedge_count, avg_weight, hausdorff_dimension, r_squared, all_dimensions}, llm: {provider, model, temperature, status}, docker: {neo4j, qdrant}, qdrant: {collection_count, memory_count}, system: {uptime, tool_count}, metabolic: {pending_traces, trace_threshold, deep_sleep_active}}` |
| `student_status` | `{current_project, adapter_path, base_model, dimension, projects[], deep_sleep_active, pending_traces}` |
| `model_info` | `{current_provider, providers: {...}}` |
| `edge_event` | `{type, source, target, weight, timestamp}` |

Frontend emits:
- `chat_message` → `{message: string}`
- `request_full_graph` → (no payload)
- `get_health_stats` → `{calculate_hausdorff: boolean}`
- `get_available_models` → (no payload)
- `request_student_status` → (no payload)
- `switch_model` → `{provider: string}`
- `switch_project` → `{project_id: string}`

## Page Status

### Chat (/) — WORKING
- Full send/receive with backend
- Thinking indicator, timestamps, keyboard shortcuts
- Auto-scroll, auto-resize textarea

### Graph (/graph) — WORKING
- Radial concentric ring layout matching original consciousness-dashboard.html
- Core nodes (Self, Working Memory, Long Term Memory, Tools, Projects) in center triangle
- Other nodes in rings by BFS distance from core
- Curved arc links between nodes
- Hyperedge visualization (Catmull-Rom closed curves, dashed stroke)
- Hover: dims unconnected nodes/edges, brightens connected paths, highlights hyperedges with glow
- Zoom/pan via D3 zoom
- No force simulation — static positions, no jitter

### Galaxy (/galaxy) — WORKING, NEEDS POLISH
- D3 force simulation computes 2D layout (200 sync ticks), mapped to 3D (x,z plane + y spread)
- Three.js rendering with UnrealBloomPass, Reinhard tone mapping
- Nodes: emissive spheres with double glow corona (additive blending) + inner hot core
- Edges: simple THREE.Line, color-coded by strength (gold >0.7, cyan mid, blue <0.3)
- Edge opacity pulsing (sine wave, phase-offset per edge)
- Node glow breathing on wide outer layer
- Labels: white canvas-texture sprites, billboarded
- Drag: raycaster hit-test, feeds fx/fy back into d3 simulation, edges update live
- Auto-rotate with orbit controls
- **Known issue**: label pulsing visual artifact — the node glow breathing (children[1] opacity animation) may still visually affect nearby labels due to additive blending. The animation targets the 5x-radius glow layer but bloom post-processing amplifies it. Possible fix: disable node breathing entirely, or render labels in a separate composer pass.

### Heatmap (/heatmap) — FUNCTIONAL
- Canvas grid, heat score per node (connections + weight)
- Click to select, shows stats
- Depends on graph data arriving via socket

### Matrix (/matrix) — FUNCTIONAL
- BFS shortest paths, canvas rendering
- Color gradient (hot → cold), diameter/density stats
- Depends on graph data

### Fractal (/fractal) — FUNCTIONAL
- Subgraph extraction at radii 1-5, D3 force mini-graphs
- Linear regression for dimension estimate
- Seed node selector
- Subscription moved to onMount (was broken at module scope)

### Health (/health) — FUNCTIONAL
- MetricCards for concepts, relationships, hyperedges, LLM, Docker, Hausdorff, traces
- Trace progress bar with threshold
- Deep sleep induction button (POST /induce_sleep)
- Hausdorff dimension from graph.hausdorff_dimension (inside graph stats, not separate key)
- Docker status checks for 'Running' not 'healthy'

### Student (/student) — FUNCTIONAL
- Own socket event (request_student_status → student_status), not from health_stats
- Model info, adapter path, project, pending traces
- Deep sleep button, merge/export placeholders

## Common Pitfalls Encountered

1. **Module-scope subscriptions** — Svelte store subscriptions at module scope fire before `onMount`, when DOM refs (`container`, etc.) are still undefined. Always subscribe inside `onMount`.

2. **ResizeObserver loops** — Modifying DOM inside a ResizeObserver callback (e.g., clearing and rebuilding SVG) can trigger the observer again. Either don't use ResizeObserver for full rebuilds, or use a re-entry guard.

3. **D3 + Three.js typing** — D3's `.call(drag)` generic signatures don't unify with selection generics. Use `// @ts-ignore` or restructure to avoid `.call()`.

4. **Three.js cylinder edges** — Oriented cylinders for edges break when parent groups rotate. Simple `THREE.Line` with `BufferGeometry.setFromPoints` always works.

5. **Backend data format** — Nodes use `label` not `name`, edges use `strength` not `weight`, chat expects `{message: text}` not raw string, Docker status is `'Running'` not `'healthy'`.

## What's Left

- [ ] Galaxy: resolve label/glow visual bleed (separate render pass or disable breathing)
- [ ] Heatmap/Matrix/Fractal: verify with live data, add empty-state indicators
- [ ] Health: add all_dimensions display (5 Hausdorff methods), auto-refresh option
- [ ] Student: project switching UI, merge/export functionality
- [ ] Mobile responsive layout (sidebar collapse)
- [ ] Edge event animations (real-time created/pruned/strengthened visual feedback)
- [ ] Voice integration (port elder-voice-apple.py)
