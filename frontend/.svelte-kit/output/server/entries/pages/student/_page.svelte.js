import { s as store_get, a9 as head, b as attr_class, a8 as stringify, aa as attr_style, c as escape_html, a as attr, u as unsubscribe_stores } from "../../../chunks/index2.js";
import { w as writable } from "../../../chunks/index.js";
import { M as MetricCard, S as SLEEP } from "../../../chunks/MetricCard.js";
import "../../../chunks/socket.js";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let traceCount, tracePercent;
    const student = writable({});
    let activePanel = null;
    traceCount = store_get($$store_subs ??= {}, "$student", student).pending_traces ?? 0;
    tracePercent = Math.min(100, traceCount / SLEEP.traceThreshold * 100);
    store_get($$store_subs ??= {}, "$student", student).adapter_path ?? "—";
    head("uo0lw", $$renderer2, ($$renderer3) => {
      $$renderer3.title(($$renderer4) => {
        $$renderer4.push(`<title>ELDER2 — Student</title>`);
      });
    });
    $$renderer2.push(`<div class="page svelte-uo0lw"><div class="page-header svelte-uo0lw"><h1 class="svelte-uo0lw">Student Model</h1></div> <div class="grid svelte-uo0lw"><div${attr_class("card-clickable svelte-uo0lw", void 0, { "active": activePanel === "lora" })}>`);
    MetricCard($$renderer2, {
      title: "LoRA Name",
      value: store_get($$store_subs ??= {}, "$student", student).current_project ?? "default"
    });
    $$renderer2.push(`<!----></div>  <div${attr_class("card-clickable svelte-uo0lw", void 0, { "active": activePanel === "model" })}>`);
    MetricCard($$renderer2, {
      title: "Model",
      value: store_get($$store_subs ??= {}, "$student", student).base_model ?? "Not configured"
    });
    $$renderer2.push(`<!----></div> `);
    MetricCard($$renderer2, {
      title: "Status",
      value: store_get($$store_subs ??= {}, "$student", student).base_model ? "Loaded" : "Idle",
      status: store_get($$store_subs ??= {}, "$student", student).base_model ? "ok" : "warn"
    });
    $$renderer2.push(`<!----> `);
    MetricCard($$renderer2, {
      title: "Pending Traces",
      value: traceCount,
      subtitle: `${stringify(tracePercent.toFixed(0))}% to threshold`
    });
    $$renderer2.push(`<!----></div> <div class="card actions-card svelte-uo0lw"><div class="card-header">Training Controls</div> <div class="trace-bar-container svelte-uo0lw"><div class="trace-bar svelte-uo0lw"><div class="trace-bar-fill svelte-uo0lw"${attr_style(`width: ${stringify(tracePercent)}%`)}></div></div> <span class="trace-label svelte-uo0lw">${escape_html(traceCount)} / ${escape_html(SLEEP.traceThreshold)} traces</span></div> <div class="btn-row svelte-uo0lw"><button class="btn-primary svelte-uo0lw"${attr("disabled", traceCount === 0, true)}>${escape_html("Induce Deep Sleep")}</button> <button class="btn-secondary svelte-uo0lw" disabled="">Merge to Base (coming soon)</button> <button class="btn-secondary svelte-uo0lw" disabled="">Export Adapter (coming soon)</button></div></div> `);
    {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> `);
    {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
