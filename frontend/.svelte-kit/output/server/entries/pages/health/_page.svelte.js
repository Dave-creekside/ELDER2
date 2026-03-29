import { s as store_get, a9 as head, a as attr, c as escape_html, a8 as stringify, aa as attr_style, u as unsubscribe_stores } from "../../../chunks/index2.js";
import { M as MetricCard, S as SLEEP } from "../../../chunks/MetricCard.js";
import { w as writable } from "../../../chunks/index.js";
import "../../../chunks/socket.js";
const healthStats = writable({});
const isRefreshing = writable(false);
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let graph, llm, docker, metabolic, traceCount, tracePercent;
    graph = store_get($$store_subs ??= {}, "$healthStats", healthStats).graph ?? {};
    llm = store_get($$store_subs ??= {}, "$healthStats", healthStats).llm ?? {};
    docker = store_get($$store_subs ??= {}, "$healthStats", healthStats).docker ?? {};
    metabolic = store_get($$store_subs ??= {}, "$healthStats", healthStats).metabolic ?? {};
    traceCount = metabolic.pending_traces ?? 0;
    tracePercent = Math.min(100, traceCount / SLEEP.traceThreshold * 100);
    head("3hm2cj", $$renderer2, ($$renderer3) => {
      $$renderer3.title(($$renderer4) => {
        $$renderer4.push(`<title>ELDER2 — Health</title>`);
      });
    });
    $$renderer2.push(`<div class="page svelte-3hm2cj"><div class="page-header svelte-3hm2cj"><h1 class="svelte-3hm2cj">System Health</h1> <button class="btn-refresh svelte-3hm2cj"${attr("disabled", store_get($$store_subs ??= {}, "$isRefreshing", isRefreshing), true)}>${escape_html(store_get($$store_subs ??= {}, "$isRefreshing", isRefreshing) ? "Refreshing..." : "Refresh")}</button></div> <div class="grid svelte-3hm2cj">`);
    MetricCard($$renderer2, {
      title: "Concepts",
      value: graph.node_count ?? "—",
      subtitle: "in Neo4j hypergraph"
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "Relationships",
      value: graph.edge_count ?? "—",
      subtitle: `avg weight: ${stringify((graph.avg_weight ?? 0).toFixed(3))}`
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, { title: "Hyperedges", value: graph.hyperedge_count ?? "—" });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "LLM",
      value: llm.model ?? "—",
      subtitle: llm.provider ?? "",
      status: llm.status === "Ready" ? "ok" : "warn"
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "Neo4j",
      value: docker.neo4j ?? "unknown",
      status: docker.neo4j === "Running" ? "ok" : "err"
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "Qdrant",
      value: docker.qdrant ?? "unknown",
      status: docker.qdrant === "Running" ? "ok" : "err"
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "Pending Traces",
      value: metabolic.pending_traces ?? 0,
      subtitle: "metabolic trace buffer"
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "Hausdorff Dimension",
      value: graph.hausdorff_dimension != null ? graph.hausdorff_dimension.toFixed(4) : "—",
      subtitle: graph.r_squared != null ? `R² = ${graph.r_squared.toFixed(4)}` : ""
    });
    $$renderer2.push(`<!----></div> <div class="sleep-section card svelte-3hm2cj"><div class="card-header">Deep Sleep Engine</div> <div class="trace-bar-container svelte-3hm2cj"><div class="trace-bar-label svelte-3hm2cj"><span>Pending Traces</span> <span class="trace-count svelte-3hm2cj">${escape_html(traceCount)} / ${escape_html(SLEEP.traceThreshold)}</span></div> <div class="trace-bar svelte-3hm2cj"><div class="trace-bar-fill svelte-3hm2cj"${attr_style(`width: ${stringify(tracePercent)}%`)}></div></div></div> <div class="sleep-actions svelte-3hm2cj"><button class="btn-sleep svelte-3hm2cj"${attr("disabled", traceCount === 0, true)}>${escape_html("Induce Deep Sleep")}</button> <span class="sleep-hint svelte-3hm2cj">`);
    if (traceCount === 0) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`No traces to consolidate`);
    } else if (traceCount < SLEEP.traceThreshold) {
      $$renderer2.push("<!--[1-->");
      $$renderer2.push(`${escape_html(SLEEP.traceThreshold - traceCount)} more traces until auto-trigger`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`Threshold reached — ready for consolidation`);
    }
    $$renderer2.push(`<!--]--></span></div></div></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
