<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { connectionStatus } from '$lib/stores/socket';
  import { initGraphStore, nodeCount, edgeCount } from '$lib/stores/graph';
  import { initChatStore } from '$lib/stores/chat';
  import { initHealthStore } from '$lib/stores/health';
  import { getSocket } from '$lib/stores/socket';
  import '../app.css';

  const NAV = [
    { href: '/',         icon: '💬', label: 'Chat' },
    { href: '/graph',    icon: '🕸️', label: 'Graph' },
    { href: '/galaxy',   icon: '🌌', label: 'Galaxy' },
    { href: '/heatmap',  icon: '🔥', label: 'Heatmap' },
    { href: '/matrix',   icon: '📊', label: 'Matrix' },
    { href: '/fractal',  icon: '🔬', label: 'Fractal' },
    { href: '/health',   icon: '🩺', label: 'Health' },
    { href: '/student',  icon: '🎓', label: 'Student' },
  ];

  onMount(() => {
    getSocket();       // establish connection
    initGraphStore();
    initChatStore();
    initHealthStore();
  });
</script>

<div class="shell">
  <nav class="sidebar">
    <div class="sidebar-brand">
      <span class="brand-icon">🧠</span>
      <span class="brand-text">ELDER<span class="brand-ver">2</span></span>
    </div>

    <div class="sidebar-nav">
      {#each NAV as item}
        <a
          href={item.href}
          class="nav-link"
          class:active={$page.url.pathname === item.href}
        >
          <span class="nav-icon">{item.icon}</span>
          <span class="nav-label">{item.label}</span>
        </a>
      {/each}
    </div>

    <div class="sidebar-footer">
      <div class="stat-row">
        <span class="stat-label">Nodes</span>
        <span class="stat-value">{$nodeCount}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Edges</span>
        <span class="stat-value">{$edgeCount}</span>
      </div>
      <div class="status-row">
        <span
          class="status-dot"
          class:connected={$connectionStatus === 'connected'}
          class:connecting={$connectionStatus === 'connecting'}
        ></span>
        <span class="status-text">{$connectionStatus}</span>
      </div>
    </div>
  </nav>

  <main class="content">
    <slot />
  </main>
</div>

<style>
  .shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── sidebar ─────────────────────────────────────────────── */
  .sidebar {
    width: var(--sidebar-width);
    min-width: var(--sidebar-width);
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 0.75rem;
    gap: 0.5rem;
  }

  .sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.25rem;
  }

  .brand-icon { font-size: 1.5rem; }
  .brand-text {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-primary);
  }
  .brand-ver {
    color: var(--accent);
    font-size: 0.85rem;
  }

  .sidebar-nav {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .nav-link {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.6rem;
    border-radius: var(--radius-sm);
    text-decoration: none;
    color: var(--text-secondary);
    font-size: 0.85rem;
    transition: all 0.15s;
  }

  .nav-link:hover {
    background: var(--bg-raised);
    color: var(--text-primary);
  }

  .nav-link.active {
    background: var(--accent-dim);
    color: var(--accent);
  }

  .nav-icon { font-size: 1rem; width: 1.4rem; text-align: center; }
  .nav-label { font-weight: 500; }

  /* ── sidebar footer ──────────────────────────────────────── */
  .sidebar-footer {
    border-top: 1px solid var(--border);
    padding-top: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    font-size: 0.9rem;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0 0.5rem;
  }

  .stat-label { color: var(--text-muted); font-size: 0.85rem; }
  .stat-value { color: var(--text-secondary); font-family: var(--font-mono); font-size: 0.95rem; font-weight: 600; }

  .status-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.5rem;
    margin-top: 0.25rem;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--error);
    flex-shrink: 0;
  }

  .status-dot.connected { background: var(--accent); }
  .status-dot.connecting { background: var(--warning); animation: pulse 1.2s infinite; }

  .status-text {
    color: var(--text-muted);
    text-transform: capitalize;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* ── main content ────────────────────────────────────────── */
  .content {
    flex: 1;
    overflow: auto;
    position: relative;
  }
</style>
