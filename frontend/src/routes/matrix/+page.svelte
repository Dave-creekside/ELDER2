<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphData, nodeCount } from '$lib/stores/graph';
  import { lerpColor } from '$lib/utils/colors';

  let canvas: HTMLCanvasElement;
  let container: HTMLDivElement;
  let stats = { diameter: 0, density: 0 };

  function render(data: typeof $graphData) {
    if (!canvas || !container || data.nodes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nodes = data.nodes;
    const n = nodes.length;
    const nameMap = new Map(nodes.map((nd, i) => [nd.id, i]));

    // BFS shortest paths
    const adj = Array.from({ length: n }, () => [] as number[]);
    data.edges.forEach((e) => {
      const si = nameMap.get(e.source);
      const ti = nameMap.get(e.target);
      if (si != null && ti != null) {
        adj[si].push(ti);
        adj[ti].push(si);
      }
    });

    const dist = Array.from({ length: n }, () => new Float32Array(n).fill(Infinity));
    for (let i = 0; i < n; i++) {
      dist[i][i] = 0;
      const queue = [i];
      let head = 0;
      while (head < queue.length) {
        const cur = queue[head++];
        for (const nb of adj[cur]) {
          if (dist[i][nb] === Infinity) {
            dist[i][nb] = dist[i][cur] + 1;
            queue.push(nb);
          }
        }
      }
    }

    // Stats
    let maxDist = 0;
    let reachablePairs = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (dist[i][j] < Infinity) {
          maxDist = Math.max(maxDist, dist[i][j]);
          reachablePairs++;
        }
      }
    }
    const totalPairs = (n * (n - 1)) / 2;
    stats = { diameter: maxDist, density: totalPairs > 0 ? reachablePairs / totalPairs : 0 };

    // Draw matrix
    const rect = container.getBoundingClientRect();
    const margin = 80;
    const size = Math.min(rect.width, rect.height - 60) - margin * 2;
    const cellSize = Math.max(2, size / n);

    canvas.width = rect.width;
    canvas.height = rect.height - 60;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const d = dist[i][j];
        let color: string;
        if (i === j) {
          color = '#1a1a2e';
        } else if (d === Infinity) {
          color = '#0a0a0f';
        } else {
          const t = maxDist > 0 ? d / maxDist : 0;
          color = lerpColor('#ff4400', '#0044ff', t);
        }
        ctx.fillStyle = color;
        ctx.fillRect(margin + j * cellSize, margin + i * cellSize, cellSize - 0.5, cellSize - 0.5);
      }
    }

    // Labels (if space allows)
    if (cellSize > 10) {
      ctx.fillStyle = 'var(--text-muted)';
      ctx.font = `${Math.min(9, cellSize * 0.7)}px sans-serif`;
      nodes.forEach((nd, i) => {
        const label = nd.label.length > 8 ? nd.label.slice(0, 7) + '…' : nd.label;
        ctx.save();
        ctx.translate(margin + i * cellSize + cellSize / 2, margin - 4);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'left';
        ctx.fillText(label, 0, 0);
        ctx.restore();
      });
    }
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

<svelte:head><title>ELDER2 — Distance Matrix</title></svelte:head>

<div class="page" bind:this={container}>
  <div class="header">
    <h1>Distance Matrix</h1>
    <span class="stat">Diameter: {stats.diameter}</span>
    <span class="stat">Density: {(stats.density * 100).toFixed(1)}%</span>
    <span class="subtitle">{$nodeCount} nodes</span>
  </div>
  <canvas bind:this={canvas}></canvas>
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

  .stat {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }

  canvas { flex: 1; }
</style>
