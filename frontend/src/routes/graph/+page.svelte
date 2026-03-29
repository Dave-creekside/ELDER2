<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import { graphData, type GraphNode, type GraphEdge, type Hyperedge } from '$lib/stores/graph';
  import { CORE_NODES, GRAPH } from '$lib/config';
  import { NODE_COLORS, CORE_NODE_COLOR, edgeOpacity } from '$lib/utils/colors';
  import * as d3 from 'd3';

  let container: HTMLDivElement;
  let lastNodeCount = -1;

  interface PosNode {
    id: string;
    label: string;
    isCore: boolean;
    color: string;
    x: number;
    y: number;
    distance: number;
  }

  function getNodeColor(id: string): string {
    let hash = 0;
    for (let i = 0; i < id.length; i++) hash = id.charCodeAt(i) + ((hash << 5) - hash);
    return NODE_COLORS[Math.abs(hash) % NODE_COLORS.length];
  }

  function buildGraph(data: { nodes: GraphNode[]; edges: GraphEdge[]; hyperedges: Hyperedge[] }) {
    if (!container || !data.nodes.length) return;

    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    if (width === 0 || height === 0) return;

    const cx = width / 2;
    const cy = height / 2;

    d3.select(container).selectAll('*').remove();

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g');
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) =>
          g.attr('transform', event.transform.toString()))
    );

    const linkGroup = g.append('g').attr('class', 'links');
    const hyperedgeGroup = g.append('g').attr('class', 'hyperedges');
    const nodeGroup = g.append('g').attr('class', 'nodes');

    // Build adjacency for BFS
    const adj = new Map<string, string[]>();
    data.edges.forEach((e) => {
      if (!adj.has(e.source)) adj.set(e.source, []);
      if (!adj.has(e.target)) adj.set(e.target, []);
      adj.get(e.source)!.push(e.target);
      adj.get(e.target)!.push(e.source);
    });

    // BFS from core nodes to get distance
    const coreIds = data.nodes
      .filter((n) => CORE_NODES.includes(n.label as any))
      .map((n) => n.id);

    const distances = new Map<string, number>();
    const queue: string[] = [...coreIds];
    coreIds.forEach((id) => distances.set(id, 0));

    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentDist = distances.get(current)!;
      for (const nb of adj.get(current) ?? []) {
        if (!distances.has(nb)) {
          distances.set(nb, currentDist + 1);
          queue.push(nb);
        }
      }
    }

    // Build positioned nodes
    const nodes: PosNode[] = data.nodes.map((n) => {
      const isCore = CORE_NODES.includes(n.label as any);
      const dist = distances.get(n.id) ?? 999;
      return {
        id: n.id,
        label: n.label,
        isCore,
        color: isCore ? CORE_NODE_COLOR : getNodeColor(n.id),
        x: 0, y: 0,
        distance: dist === 999 ? 4 : dist,
      };
    });

    // Group by distance ring
    const byDistance = new Map<number, PosNode[]>();
    nodes.forEach((n) => {
      if (!byDistance.has(n.distance)) byDistance.set(n.distance, []);
      byDistance.get(n.distance)!.push(n);
    });

    // Position: core in center triangle, others in concentric rings
    byDistance.forEach((ring, dist) => {
      if (dist === 0) {
        ring.forEach((n, i) => {
          const angle = (i * 2 * Math.PI) / Math.max(ring.length, 3) - Math.PI / 2;
          n.x = cx + Math.cos(angle) * 30;
          n.y = cy + Math.sin(angle) * 30;
        });
      } else {
        const radius = 80 + Math.min(dist, 5) * 70;
        const step = (2 * Math.PI) / ring.length;
        ring.forEach((n, i) => {
          const angle = i * step - Math.PI / 2;
          n.x = cx + Math.cos(angle) * radius;
          n.y = cy + Math.sin(angle) * radius;
        });
      }
    });

    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    // Draw curved links
    linkGroup.selectAll('path')
      .data(data.edges)
      .join('path')
      .attr('fill', 'none')
      .attr('stroke', CORE_NODE_COLOR)
      .attr('stroke-opacity', (d) => edgeOpacity(d.strength))
      .attr('stroke-width', (d) => 0.5 + d.strength * 1.5)
      .attr('d', (d) => {
        const s = nodeMap.get(d.source);
        const t = nodeMap.get(d.target);
        if (!s || !t) return null;
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const dr = Math.sqrt(dx * dx + dy * dy) * 0.3;
        return `M${s.x},${s.y}A${dr},${dr} 0 0,1 ${t.x},${t.y}`;
      });

    // Draw hyperedges as closed curves
    if (data.hyperedges.length > 0) {
      const curve = d3.line<PosNode>()
        .x((d) => d.x)
        .y((d) => d.y)
        .curve(d3.curveCatmullRomClosed);

      hyperedgeGroup.selectAll('path')
        .data(data.hyperedges.filter((he) => he.members.length >= 2))
        .join('path')
        .attr('fill', 'rgba(0, 255, 136, 0.04)')
        .attr('stroke', 'rgba(0, 255, 136, 0.15)')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,4')
        .attr('d', (he) => {
          const members = he.members.map((id) => nodeMap.get(id)).filter(Boolean) as PosNode[];
          if (members.length < 2) return null;
          const hcx = d3.mean(members, (m) => m.x) ?? 0;
          const hcy = d3.mean(members, (m) => m.y) ?? 0;
          members.sort((a, b) =>
            Math.atan2(a.y - hcy, a.x - hcx) - Math.atan2(b.y - hcy, b.x - hcx)
          );
          return curve(members);
        });
    }

    // Draw nodes
    const nodeEl = nodeGroup.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', (d) => `translate(${d.x},${d.y})`)
      .style('cursor', 'pointer');

    // Glow ring
    nodeEl.append('circle')
      .attr('r', (d) => d.isCore ? 18 : 12)
      .attr('fill', 'none')
      .attr('stroke', (d) => d.color)
      .attr('stroke-opacity', 0.15)
      .attr('stroke-width', 4);

    // Main circle
    nodeEl.append('circle')
      .attr('r', (d) => d.isCore ? 10 : 6)
      .attr('fill', (d) => d.color)
      .attr('stroke', 'rgba(255,255,255,0.4)')
      .attr('stroke-width', 1.5)
      .style('filter', (d) => `drop-shadow(0 0 ${d.isCore ? 12 : 6}px ${d.color})`);

    // Inner highlight
    nodeEl.append('circle')
      .attr('r', (d) => d.isCore ? 4 : 2.5)
      .attr('fill', 'rgba(255,255,255,0.3)');

    // Labels
    nodeEl.append('text')
      .attr('dy', (d) => d.isCore ? -16 : -12)
      .attr('text-anchor', 'middle')
      .attr('font-size', (d) => d.isCore ? '11px' : '9px')
      .attr('font-weight', (d) => d.isCore ? '600' : '400')
      .attr('fill', 'rgba(255,255,255,0.85)')
      .style('text-shadow', '0 0 8px rgba(0,0,0,0.8)')
      .text((d) => d.label);

    // Hover interaction — highlight connected nodes and their edges
    nodeEl.on('mouseover', function (event, d) {
      const connected = new Set<string>();
      connected.add(d.id);

      // Collect connected edge indices
      const connectedEdges = new Set<number>();
      data.edges.forEach((e, i) => {
        if (e.source === d.id) { connected.add(e.target); connectedEdges.add(i); }
        if (e.target === d.id) { connected.add(e.source); connectedEdges.add(i); }
      });

      // Brighten hovered node glow
      d3.select(this).select('circle:nth-child(1)')
        .transition().duration(200)
        .attr('stroke-opacity', 0.7)
        .attr('stroke-width', 6);
      d3.select(this).select('circle:nth-child(2)')
        .transition().duration(200)
        .style('filter', `drop-shadow(0 0 20px ${d.color})`);

      // Dim/brighten nodes
      nodeGroup.selectAll<SVGGElement, PosNode>('g')
        .transition().duration(200)
        .style('opacity', (n) => connected.has(n.id) ? 1 : 0.1);

      // Dim/brighten edges
      linkGroup.selectAll<SVGPathElement, GraphEdge>('path')
        .transition().duration(200)
        .attr('stroke-opacity', (_, i) => connectedEdges.has(i) ? 0.8 : 0.03)
        .attr('stroke-width', (e, i) => connectedEdges.has(i) ? 1.5 + e.strength * 3 : 0.5)
        .attr('stroke', (_, i) => connectedEdges.has(i) ? '#00ff88' : '#00ff88');

      // Highlight hyperedges containing this node, dim the rest
      hyperedgeGroup.selectAll('path')
        .transition().duration(200)
        .attr('fill', (he: any) =>
          he.members?.includes(d.id) ? 'rgba(0, 255, 136, 0.12)' : 'rgba(0, 255, 136, 0.01)')
        .attr('stroke', (he: any) =>
          he.members?.includes(d.id) ? 'rgba(0, 255, 136, 0.7)' : 'rgba(0, 255, 136, 0.03)')
        .attr('stroke-width', (he: any) =>
          he.members?.includes(d.id) ? 2.5 : 0.5)
        .attr('stroke-dasharray', (he: any) =>
          he.members?.includes(d.id) ? 'none' : '4,4')
        .style('filter', (he: any) =>
          he.members?.includes(d.id) ? 'drop-shadow(0 0 8px rgba(0, 255, 136, 0.4))' : 'none');

    }).on('mouseout', function (event, d) {
      // Restore glow
      d3.select(this).select('circle:nth-child(1)')
        .transition().duration(300)
        .attr('stroke-opacity', 0.15)
        .attr('stroke-width', 4);
      d3.select(this).select('circle:nth-child(2)')
        .transition().duration(300)
        .style('filter', `drop-shadow(0 0 ${d.isCore ? 12 : 6}px ${d.color})`);

      // Restore all nodes
      nodeGroup.selectAll('g')
        .transition().duration(300)
        .style('opacity', 1);

      // Restore all edges
      linkGroup.selectAll<SVGPathElement, GraphEdge>('path')
        .transition().duration(300)
        .attr('stroke-opacity', (e) => edgeOpacity(e.strength))
        .attr('stroke-width', (e) => 0.5 + e.strength * 1.5);

      // Restore hyperedges
      hyperedgeGroup.selectAll('path')
        .transition().duration(300)
        .attr('fill', 'rgba(0, 255, 136, 0.04)')
        .attr('stroke', 'rgba(0, 255, 136, 0.15)')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,4')
        .style('filter', 'none');
    });

    lastNodeCount = data.nodes.length;
  }

  let unsubscribe: () => void;
  let building = false;

  onMount(() => {
    unsubscribe = graphData.subscribe((data) => {
      if (container && data.nodes.length > 0 && data.nodes.length !== lastNodeCount && !building) {
        building = true;
        // Defer to next frame so the DOM is stable
        requestAnimationFrame(() => {
          buildGraph(data);
          building = false;
        });
      }
    });
  });

  onDestroy(() => {
    if (unsubscribe) unsubscribe();
  });
</script>

<svelte:head><title>ELDER2 — Graph</title></svelte:head>

<div class="page" bind:this={container}></div>

<style>
  .page {
    height: 100%;
    position: relative;
    background:
      radial-gradient(ellipse at 20% 30%, rgba(0, 100, 150, 0.15) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 70%, rgba(100, 0, 150, 0.1) 0%, transparent 50%),
      radial-gradient(ellipse at 50% 50%, rgba(0, 50, 100, 0.2) 0%, transparent 70%),
      radial-gradient(ellipse at center, #0a0e18 0%, #030508 100%);
  }
</style>
