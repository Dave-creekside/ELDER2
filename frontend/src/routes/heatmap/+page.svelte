<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphData, nodeCount, edgeCount } from '$lib/stores/graph';
  import { heatColor } from '$lib/utils/colors';

  let canvas: HTMLCanvasElement;
  let container: HTMLDivElement;
  let selected: { label: string; heat: number; connections: number } | null = null;

  interface CellData {
    label: string;
    heat: number;
    connections: number;
    weight: number;
    x: number;
    y: number;
    w: number;
    h: number;
  }

  let cells: CellData[] = [];

  function render(data: typeof $graphData) {
    if (!canvas || !container || data.nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height - 60; // leave space for header

    // Compute heat per node
    const connectionCounts = new Map<string, number>();
    const weightSums = new Map<string, number>();

    data.edges.forEach((e) => {
      connectionCounts.set(e.source, (connectionCounts.get(e.source) ?? 0) + 1);
      connectionCounts.set(e.target, (connectionCounts.get(e.target) ?? 0) + 1);
      const w = e.strength;
      weightSums.set(e.source, (weightSums.get(e.source) ?? 0) + w);
      weightSums.set(e.target, (weightSums.get(e.target) ?? 0) + w);
    });

    const maxConn = Math.max(1, ...connectionCounts.values());

    const nodeHeats = data.nodes.map((n) => {
      const id = n.id;
      const conn = connectionCounts.get(id) ?? 0;
      const wt = weightSums.get(id) ?? 0;
      const heat = 0.5 * (conn / maxConn) + 0.5 * Math.min(wt / Math.max(conn, 1), 1);
      return { label: n.label, heat, connections: conn, weight: wt };
    });

    nodeHeats.sort((a, b) => b.heat - a.heat);

    // Grid layout
    const cols = Math.ceil(Math.sqrt(nodeHeats.length));
    const cellW = canvas.width / cols;
    const cellH = cellW; // square cells

    cells = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    nodeHeats.forEach((node, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = col * cellW;
      const y = row * cellH;

      cells.push({ ...node, x, y, w: cellW, h: cellH });

      ctx.fillStyle = heatColor(node.heat);
      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

      // Label if cell large enough
      if (cellW > 40) {
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = `${Math.min(10, cellW / 6)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(
          node.label.length > 12 ? node.label.slice(0, 11) + '…' : node.label,
          x + cellW / 2,
          y + cellH / 2 + 3
        );
      }
    });
  }

  function handleClick(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const hit = cells.find((c) => mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h);
    selected = hit ? { label: hit.label, heat: hit.heat, connections: hit.connections } : null;
  }

  const unsubscribe = graphData.subscribe((data) => render(data));

  onMount(() => {
    const ro = new ResizeObserver(() => {
      let snap: typeof $graphData = { nodes: [], edges: [], hyperedges: [] };
      graphData.subscribe((v) => (snap = v))();
      render(snap);
    });
    ro.observe(container);
    return () => ro.disconnect();
  });

  onDestroy(unsubscribe);
</script>

<svelte:head><title>ELDER2 — Heatmap</title></svelte:head>

<div class="page" bind:this={container}>
  <div class="header">
    <h1>Dimensional Heatmap</h1>
    <span class="subtitle">{$nodeCount} nodes &middot; {$edgeCount} edges</span>
    {#if selected}
      <span class="selected-info">
        {selected.label} — heat: {selected.heat.toFixed(3)}, connections: {selected.connections}
      </span>
    {/if}
  </div>
  <canvas bind:this={canvas} on:click={handleClick}></canvas>
</div>

<style>
  .page {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
  }

  .header {
    padding: 0.75rem 1rem;
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-shrink: 0;
  }

  h1 { font-size: 1rem; font-weight: 600; }
  .subtitle { font-size: 0.8rem; color: var(--text-muted); }

  .selected-info {
    font-size: 0.8rem;
    color: var(--accent);
    font-family: var(--font-mono);
    margin-left: auto;
  }

  canvas {
    flex: 1;
    cursor: crosshair;
  }
</style>
