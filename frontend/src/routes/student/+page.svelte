<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import MetricCard from '$lib/components/MetricCard.svelte';
  import { induceDeepSleep } from '$lib/stores/health';
  import { getSocket } from '$lib/stores/socket';
  import { SLEEP } from '$lib/config';

  interface StudentStatus {
    current_project?: string;
    adapter_path?: string;
    base_model?: string;
    dimension?: number | string;
    projects?: string[];
    deep_sleep_active?: boolean;
    pending_traces?: number;
  }

  const student = writable<StudentStatus>({});
  let sleepInProgress = false;

  // Panel state: null, 'lora', or 'model'
  let activePanel: 'lora' | 'model' | null = null;

  // Edit fields
  let loraNameInput = '';
  let modelInput = '';
  let saving = false;
  let saveMessage = '';

  onMount(() => {
    const socket = getSocket();
    socket.on('student_status', (data: StudentStatus) => {
      student.set(data);
      // Sync inputs with current values
      if (!loraNameInput) loraNameInput = data.current_project ?? 'default';
      if (!modelInput) modelInput = data.base_model ?? '';
    });
    socket.on('project_switched', (data: { success: boolean; project: string }) => {
      saving = false;
      if (data.success) {
        saveMessage = `Switched to "${data.project}"`;
        setTimeout(() => saveMessage = '', 3000);
      }
    });
    socket.on('model_switch_status', (data: { status: string; model_id: string; error?: string }) => {
      if (data.status === 'loading') {
        saving = true;
        saveMessage = `Loading ${data.model_id}...`;
      } else if (data.status === 'loaded') {
        saving = false;
        saveMessage = `Loaded ${data.model_id}`;
        modelInput = data.model_id;
        setTimeout(() => saveMessage = '', 3000);
      } else if (data.status === 'error') {
        saving = false;
        saveMessage = `Error: ${data.error}`;
        setTimeout(() => saveMessage = '', 5000);
      }
    });
    socket.emit('request_student_status');
  });

  function togglePanel(panel: 'lora' | 'model') {
    if (activePanel === panel) {
      activePanel = null;
    } else {
      activePanel = panel;
      saveMessage = '';
      // Reset inputs to current values
      loraNameInput = $student.current_project ?? 'default';
      modelInput = $student.base_model ?? '';
    }
  }

  function saveLora() {
    const name = loraNameInput.trim();
    if (!name) return;
    saving = true;
    saveMessage = '';
    getSocket().emit('switch_project', { project_id: name });
  }

  function saveModel() {
    const id = modelInput.trim();
    if (!id) return;
    saving = true;
    saveMessage = '';
    getSocket().emit('switch_student_model', { model_id: id });
  }

  async function handleSleep() {
    sleepInProgress = true;
    await induceDeepSleep();
    sleepInProgress = false;
    getSocket().emit('request_student_status');
  }

  $: traceCount = $student.pending_traces ?? 0;
  $: tracePercent = Math.min(100, (traceCount / SLEEP.traceThreshold) * 100);
  $: adapterPath = $student.adapter_path ?? '—';
</script>

<svelte:head><title>ELDER2 — Student</title></svelte:head>

<div class="page">
  <div class="page-header">
    <h1>Student Model</h1>
  </div>

  <div class="grid">
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="card-clickable" class:active={activePanel === 'lora'} on:click={() => togglePanel('lora')}>
      <MetricCard
        title="LoRA Name"
        value={$student.current_project ?? 'default'}
      />
    </div>
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="card-clickable" class:active={activePanel === 'model'} on:click={() => togglePanel('model')}>
      <MetricCard
        title="Model"
        value={$student.base_model ?? 'Not configured'}
      />
    </div>
    <MetricCard
      title="Status"
      value={$student.base_model ? 'Loaded' : 'Idle'}
      status={$student.base_model ? 'ok' : 'warn'}
    />
    <MetricCard
      title="Pending Traces"
      value={traceCount}
      subtitle="{tracePercent.toFixed(0)}% to threshold"
    />
  </div>

  <div class="card actions-card">
    <div class="card-header">Training Controls</div>

    <div class="trace-bar-container">
      <div class="trace-bar">
        <div class="trace-bar-fill" style="width: {tracePercent}%"></div>
      </div>
      <span class="trace-label">{traceCount} / {SLEEP.traceThreshold} traces</span>
    </div>

    <div class="btn-row">
      <button class="btn-primary" on:click={handleSleep} disabled={sleepInProgress || traceCount === 0}>
        {sleepInProgress ? 'Running...' : 'Induce Deep Sleep'}
      </button>
      <button class="btn-secondary" disabled>Merge to Base (coming soon)</button>
      <button class="btn-secondary" disabled>Export Adapter (coming soon)</button>
    </div>
  </div>

  {#if activePanel === 'lora'}
    <div class="edit-panel">
      <div class="card-header">LoRA Configuration</div>
      <div class="edit-body">
        <label class="edit-field">
          <span class="edit-label">LoRA Name</span>
          <input
            type="text"
            bind:value={loraNameInput}
            placeholder="e.g. consciousness-v2"
            on:keydown={(e) => e.key === 'Enter' && saveLora()}
          />
          <span class="edit-hint">Adapter saves to: adapters/{loraNameInput.trim() || '...'}/</span>
        </label>
        <div class="edit-actions">
          <button class="btn-primary" on:click={saveLora} disabled={saving || !loraNameInput.trim()}>
            {saving ? 'Switching...' : 'Switch Project'}
          </button>
          {#if saveMessage}
            <span class="save-msg">{saveMessage}</span>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  {#if activePanel === 'model'}
    <div class="edit-panel">
      <div class="card-header">Student Model Configuration</div>
      <div class="edit-body">
        <label class="edit-field">
          <span class="edit-label">HuggingFace Repo or Local Path</span>
          <input
            type="text"
            bind:value={modelInput}
            placeholder="e.g. unsloth/gemma-3-4b-it or /path/to/model"
            on:keydown={(e) => e.key === 'Enter' && saveModel()}
          />
          <span class="edit-hint">Currently loaded: {$student.base_model ?? 'none'}</span>
        </label>
        <div class="edit-actions">
          <button class="btn-primary" on:click={saveModel} disabled={saving || !modelInput.trim()}>
            {saving ? 'Loading...' : 'Load Model'}
          </button>
          {#if saveMessage}
            <span class="save-msg">{saveMessage}</span>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .page {
    padding: 1.5rem;
    overflow-y: auto;
    height: 100%;
  }

  .page-header { margin-bottom: 1.25rem; }
  h1 { font-size: 1.2rem; font-weight: 600; }

  .grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
  }

  /* ── Clickable card wrappers ── */
  .card-clickable {
    cursor: pointer;
    border-radius: var(--radius-md);
    transition: box-shadow 0.15s, border-color 0.15s;
  }

  .card-clickable:hover {
    box-shadow: 0 0 0 1px var(--accent);
  }

  .card-clickable.active {
    box-shadow: 0 0 0 2px var(--accent);
  }

  .card-clickable.active :global(.metric-card) {
    border-color: var(--accent);
  }

  /* ── Training controls ── */
  .actions-card { margin-top: 0.5rem; }

  .trace-bar-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .trace-bar {
    flex: 1;
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

  .trace-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    white-space: nowrap;
  }

  .btn-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .btn-primary {
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    border: none;
    background: var(--accent);
    color: var(--bg-primary);
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
  }

  .btn-primary:disabled { opacity: 0.3; cursor: default; }

  .btn-secondary {
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: 0.85rem;
    cursor: default;
    opacity: 0.5;
  }

  /* ── Edit panels ── */
  .edit-panel {
    margin-top: 1.25rem;
    background: var(--bg-surface);
    border: 1px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1rem;
    animation: panelSlide 0.15s ease-out;
  }

  @keyframes panelSlide {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .edit-body {
    margin-top: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .edit-field {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .edit-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
  }

  .edit-field input {
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.75rem;
    font-size: 0.95rem;
    font-family: var(--font-mono);
    outline: none;
    transition: border-color 0.15s;
  }

  .edit-field input:focus {
    border-color: var(--accent);
  }

  .edit-field input::placeholder {
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 0.85rem;
  }

  .edit-hint {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .edit-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .save-msg {
    font-size: 0.8rem;
    color: var(--accent);
    font-family: var(--font-mono);
  }
</style>
