<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphData, type GraphNode, type GraphEdge } from '$lib/stores/graph';
  import { CORE_NODE_COLOR, NODE_COLORS } from '$lib/utils/colors';
  import * as d3 from 'd3';

  let gridEl: HTMLDivElement;
  let seedNode = '';
  let dimension = 0;
  let rSquared = 0;
  let nodeNames: string[] = [];

  // Per-radius stats for the side panel
  interface RadiusInfo {
    radius: number | 'full';
    nodeCount: number;
    edgeCount: number;
    logR: number;
    logN: number;
  }
  let radiusStats: RadiusInfo[] = [];

  function extractSubgraph(nodes: GraphNode[], edges: GraphEdge[], seed: string, radius: number) {
    const adj = new Map<string, string[]>();
    edges.forEach((e) => {
      if (!adj.has(e.source)) adj.set(e.source, []);
      if (!adj.has(e.target)) adj.set(e.target, []);
      adj.get(e.source)!.push(e.target);
      adj.get(e.target)!.push(e.source);
    });

    const visited = new Set<string>();
    const queue: [string, number][] = [[seed, 0]];
    visited.add(seed);

    while (queue.length > 0) {
      const [node, depth] = queue.shift()!;
      if (depth >= radius) continue;
      for (const nb of adj.get(node) ?? []) {
        if (!visited.has(nb)) {
          visited.add(nb);
          queue.push([nb, depth + 1]);
        }
      }
    }

    return {
      nodes: nodes.filter((n) => visited.has(n.id)),
      edges: edges.filter((e) => visited.has(e.source) && visited.has(e.target)),
    };
  }

  function linreg(xs: number[], ys: number[]) {
    const n = xs.length;
    if (n < 2) return { slope: 0, intercept: 0, r2: 0 };
    const mx = xs.reduce((a, b) => a + b, 0) / n;
    const my = ys.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0, ssRes = 0, ssTot = 0;
    for (let i = 0; i < n; i++) {
      num += (xs[i] - mx) * (ys[i] - my);
      den += (xs[i] - mx) ** 2;
    }
    const slope = den !== 0 ? num / den : 0;
    const intercept = my - slope * mx;
    for (let i = 0; i < n; i++) {
      const pred = slope * xs[i] + intercept;
      ssRes += (ys[i] - pred) ** 2;
      ssTot += (ys[i] - my) ** 2;
    }
    return { slope, intercept, r2: ssTot !== 0 ? 1 - ssRes / ssTot : 0 };
  }

  function compute(data: typeof $graphData) {
    if (!gridEl || data.nodes.length === 0) return;

    nodeNames = data.nodes.map((n) => n.label);
    if (!seedNode && nodeNames.length > 0) seedNode = nodeNames[0];
    if (!seedNode) return;

    const seedId = data.nodes.find((n) => n.label === seedNode)?.id ?? seedNode;

    const radii = [1, 2, 3, 4, 5];
    const logR: number[] = [];
    const logN: number[] = [];
    radiusStats = [];

    radii.forEach((r) => {
      const sub = extractSubgraph(data.nodes, data.edges, seedId, r);
      if (sub.nodes.length > 0) {
        logR.push(Math.log(r));
        logN.push(Math.log(sub.nodes.length));
      }
      radiusStats.push({
        radius: r,
        nodeCount: sub.nodes.length,
        edgeCount: sub.edges.length,
        logR: Math.log(r),
        logN: sub.nodes.length > 0 ? Math.log(sub.nodes.length) : 0,
      });
    });

    // Full graph stats
    radiusStats.push({
      radius: 'full',
      nodeCount: data.nodes.length,
      edgeCount: data.edges.length,
      logR: 0,
      logN: Math.log(data.nodes.length),
    });

    if (logR.length >= 2) {
      const reg = linreg(logR, logN);
      dimension = reg.slope;
      rSquared = reg.r2;
    }

    renderScales(data, seedId);
  }

  function renderScales(data: typeof $graphData, seedId: string) {
    if (!gridEl) return;
    // Clear previous SVGs
    gridEl.querySelectorAll('.scale-panel svg').forEach((el) => el.remove());

    const panels = gridEl.querySelectorAll('.scale-panel');
    const scales: (number | 'full')[] = [1, 2, 3, 'full'];

    panels.forEach((panel, idx) => {
      const r = scales[idx];
      const sub = r === 'full'
        ? { nodes: data.nodes, edges: data.edges }
        : extractSubgraph(data.nodes, data.edges, seedId, r as number);

      // Update label
      const label = panel.querySelector('.scale-label');
      if (label) label.textContent = r === 'full' ? `Full graph (${sub.nodes.length})` : `r = ${r}  (${sub.nodes.length} nodes)`;

      const rect = panel.getBoundingClientRect();
      const w = rect.width - 16; // padding
      const h = rect.height - 36; // label height + padding
      if (w <= 0 || h <= 0) return;

      const svg = d3.select(panel).append('svg').attr('width', w).attr('height', h);
      const g = svg.append('g');

      const simNodes = sub.nodes.map((n) => ({ id: n.id, label: n.label }));
      const nodeIds = new Set(simNodes.map((n) => n.id));
      const simLinks = sub.edges
        .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
        .map((e) => ({ source: e.source, target: e.target }));

      const sim = d3.forceSimulation(simNodes as any)
        .force('link', d3.forceLink(simLinks).id((d: any) => d.id).distance(30))
        .force('charge', d3.forceManyBody().strength(-40))
        .force('center', d3.forceCenter(w / 2, h / 2))
        .stop();

      for (let i = 0; i < 120; i++) sim.tick();

      g.selectAll('line')
        .data(simLinks)
        .join('line')
        .attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y)
        .attr('stroke', 'rgba(0,255,136,0.2)').attr('stroke-width', 1);

      g.selectAll('circle')
        .data(simNodes)
        .join('circle')
        .attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y)
        .attr('r', (d: any) => d.id === seedId ? 5 : 3)
        .attr('fill', (d: any) => d.id === seedId ? '#ffffff' : CORE_NODE_COLOR);
    });
  }

  let unsubscribe: () => void;

  onMount(() => {
    unsubscribe = graphData.subscribe((data) => compute(data));
  });

  onDestroy(() => {
    if (unsubscribe) unsubscribe();
  });
</script>

<svelte:head><title>ELDER2 — Fractal</title></svelte:head>

<div class="page">
  <div class="graph-area" bind:this={gridEl}>
    <div class="scale-panel">
      <div class="scale-label">r = 1</div>
    </div>
    <div class="scale-panel">
      <div class="scale-label">r = 2</div>
    </div>
    <div class="scale-panel">
      <div class="scale-label">r = 3</div>
    </div>
    <div class="scale-panel">
      <div class="scale-label">Full graph</div>
    </div>
  </div>

  <aside class="stats-panel">
    <label class="seed-select">
      <span class="seed-label">Seed Node</span>
      <select bind:value={seedNode} on:change={() => compute($graphData)}>
        {#each nodeNames as name}
          <option value={name}>{name}</option>
        {/each}
      </select>
    </label>

    <div class="stat-hero">
      <div class="stat-block">
        <span class="stat-key">D</span>
        <span class="stat-val">{dimension.toFixed(3)}</span>
      </div>
      <div class="stat-block">
        <span class="stat-key">R²</span>
        <span class="stat-val">{rSquared.toFixed(3)}</span>
      </div>
    </div>

    <div class="divider"></div>

    <div class="radius-list">
      <div class="radius-heading">Per-radius</div>
      {#each radiusStats as rs}
        <div class="radius-row">
          <span class="radius-label">
            {rs.radius === 'full' ? 'Full' : `r = ${rs.radius}`}
          </span>
          <div class="radius-detail">
            <span>{rs.nodeCount} nodes</span>
            <span>{rs.edgeCount} edges</span>
          </div>
          {#if rs.radius !== 'full'}
            <div class="radius-log">
              ln(r) = {rs.logR.toFixed(2)}, ln(n) = {rs.logN.toFixed(2)}
            </div>
          {/if}
        </div>
      {/each}
    </div>

    <div class="divider"></div>

    <div class="radius-list">
      <div class="radius-heading">Dimensionality</div>
      <div class="dim-row">
        <span class="dim-label">Hausdorff (fractal)</span>
        <span class="dim-val">{dimension.toFixed(4)}</span>
      </div>
      <div class="dim-row">
        <span class="dim-label">Topological</span>
        <span class="dim-val">{dimension > 0 ? Math.floor(dimension).toFixed(0) : '—'}</span>
      </div>
      <div class="dim-row">
        <span class="dim-label">Growth rate</span>
        <span class="dim-val">{dimension > 0 ? (Math.exp(dimension)).toFixed(2) + '×/r' : '—'}</span>
      </div>
      <div class="dim-row">
        <span class="dim-label">Fit quality</span>
        <span class="dim-val {rSquared > 0.9 ? 'good' : rSquared > 0.7 ? 'ok' : 'poor'}">
          {rSquared > 0.9 ? 'Strong' : rSquared > 0.7 ? 'Moderate' : 'Weak'}
        </span>
      </div>
    </div>
  </aside>
</div>

<style>
  .page {
    height: 100%;
    display: flex;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* ── Left: 2×2 graph grid, fills available space ── */
  .graph-area {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 0.75rem;
    padding: 0.75rem;
    min-width: 0;
  }

  .scale-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.5rem;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .scale-panel :global(svg) {
    flex: 1;
    width: 100%;
  }

  .scale-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
    font-family: var(--font-mono);
    flex-shrink: 0;
  }

  /* ── Right: stats panel ── */
  .stats-panel {
    width: 240px;
    min-width: 240px;
    background: var(--bg-surface);
    border-left: 1px solid var(--border);
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    overflow-y: auto;
  }

  .seed-select {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .seed-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }

  .seed-select select {
    width: 100%;
    background: var(--bg-raised);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.4rem 0.5rem;
    font-size: 0.85rem;
  }

  /* ── Hero stats ── */
  .stat-hero {
    display: flex;
    gap: 0.75rem;
  }

  .stat-block {
    flex: 1;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem;
    text-align: center;
  }

  .stat-key {
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
  }

  .stat-val {
    display: block;
    font-size: 1.3rem;
    font-weight: 700;
    font-family: var(--font-mono);
    color: var(--accent);
  }

  .divider {
    height: 1px;
    background: var(--border);
  }

  /* ── Per-radius breakdown ── */
  .radius-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .radius-heading {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }

  .radius-row {
    background: var(--bg-raised);
    border-radius: var(--radius-sm);
    padding: 0.4rem 0.5rem;
  }

  .radius-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: var(--font-mono);
  }

  .radius-detail {
    display: flex;
    gap: 0.75rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
  }

  .radius-log {
    font-size: 0.65rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-top: 0.1rem;
  }

  /* ── Dimensionality section ── */
  .dim-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
  }

  .dim-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
  }

  .dim-val {
    font-size: 0.85rem;
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--text-primary);
  }

  .dim-val.good { color: var(--accent); }
  .dim-val.ok { color: var(--warning); }
  .dim-val.poor { color: var(--error); }
</style>
