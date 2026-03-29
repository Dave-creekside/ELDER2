<script lang="ts">
  import { onMount } from 'svelte';
  import MetricCard from '$lib/components/MetricCard.svelte';
  import { healthStats, isRefreshing, requestHealth, induceDeepSleep } from '$lib/stores/health';
  import { SLEEP } from '$lib/config';

  let sleepInProgress = false;

  onMount(() => {
    requestHealth();
  });

  async function handleSleep() {
    sleepInProgress = true;
    const result = await induceDeepSleep();
    sleepInProgress = false;
    if (result.success) {
      requestHealth(); // refresh stats
    }
  }

  $: graph = $healthStats.graph ?? {};
  $: llm = $healthStats.llm ?? {};
  $: docker = $healthStats.docker ?? {};
  $: metabolic = $healthStats.metabolic ?? {};
  $: traceCount = metabolic.pending_traces ?? 0;
  $: tracePercent = Math.min(100, (traceCount / SLEEP.traceThreshold) * 100);
</script>

<svelte:head><title>ELDER2 — Health</title></svelte:head>

<div class="page">
  <div class="page-header">
    <h1>System Health</h1>
    <button class="btn-refresh" on:click={() => requestHealth()} disabled={$isRefreshing}>
      {$isRefreshing ? 'Refreshing...' : 'Refresh'}
    </button>
  </div>

  <div class="grid">
    <!-- Graph Stats -->
    <MetricCard
      title="Concepts"
      value={graph.node_count ?? '—'}
      subtitle="in Neo4j hypergraph"
    />
    <MetricCard
      title="Relationships"
      value={graph.edge_count ?? '—'}
      subtitle="avg weight: {(graph.avg_weight ?? 0).toFixed(3)}"
    />
    <MetricCard
      title="Hyperedges"
      value={graph.hyperedge_count ?? '—'}
    />

    <!-- LLM -->
    <MetricCard
      title="LLM"
      value={llm.model ?? '—'}
      subtitle={llm.provider ?? ''}
      status={llm.status === 'Ready' ? 'ok' : 'warn'}
    />

    <!-- Docker Services -->
    <MetricCard
      title="Neo4j"
      value={docker.neo4j ?? 'unknown'}
      status={docker.neo4j === 'Running' ? 'ok' : 'err'}
    />
    <MetricCard
      title="Qdrant"
      value={docker.qdrant ?? 'unknown'}
      status={docker.qdrant === 'Running' ? 'ok' : 'err'}
    />

    <!-- Metabolic -->
    <MetricCard
      title="Pending Traces"
      value={metabolic.pending_traces ?? 0}
      subtitle="metabolic trace buffer"
    />

    <!-- Hausdorff Dimension (inside graph stats from backend) -->
    <MetricCard
      title="Hausdorff Dimension"
      value={graph.hausdorff_dimension != null ? graph.hausdorff_dimension.toFixed(4) : '—'}
      subtitle={graph.r_squared != null ? `R² = ${graph.r_squared.toFixed(4)}` : ''}
    />
  </div>

  <!-- Trace accumulation + Deep Sleep -->
  <div class="sleep-section card">
    <div class="card-header">Deep Sleep Engine</div>

    <div class="trace-bar-container">
      <div class="trace-bar-label">
        <span>Pending Traces</span>
        <span class="trace-count">{traceCount} / {SLEEP.traceThreshold}</span>
      </div>
      <div class="trace-bar">
        <div class="trace-bar-fill" style="width: {tracePercent}%"></div>
      </div>
    </div>

    <div class="sleep-actions">
      <button
        class="btn-sleep"
        on:click={handleSleep}
        disabled={sleepInProgress || traceCount === 0}
      >
        {sleepInProgress ? 'Consolidating...' : 'Induce Deep Sleep'}
      </button>
      <span class="sleep-hint">
        {#if traceCount === 0}
          No traces to consolidate
        {:else if traceCount < SLEEP.traceThreshold}
          {SLEEP.traceThreshold - traceCount} more traces until auto-trigger
        {:else}
          Threshold reached — ready for consolidation
        {/if}
      </span>
    </div>
  </div>
</div>

<style>
  .page {
    padding: 1.5rem;
    overflow-y: auto;
    height: 100%;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
  }

  h1 {
    font-size: 1.2rem;
    font-weight: 600;
  }

  .btn-refresh {
    padding: 0.4rem 0.85rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--bg-raised);
    color: var(--text-primary);
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-refresh:hover:not(:disabled) { border-color: var(--accent); }
  .btn-refresh:disabled { opacity: 0.4; cursor: default; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.25rem;
  }

  /* ── sleep section ─────────────────────────────── */
  .sleep-section { margin-top: 0.5rem; }

  .trace-bar-container { margin-bottom: 1rem; }

  .trace-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.35rem;
  }

  .trace-count { font-family: var(--font-mono); }

  .trace-bar {
    height: 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
  }

  .trace-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 4px;
    transition: width 0.4s ease;
  }

  .sleep-actions {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .btn-sleep {
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    border: none;
    background: var(--accent);
    color: var(--bg-primary);
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .btn-sleep:hover:not(:disabled) { opacity: 0.85; }
  .btn-sleep:disabled { opacity: 0.3; cursor: default; }

  .sleep-hint {
    font-size: 0.8rem;
    color: var(--text-muted);
  }
</style>
