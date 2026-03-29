import { ac as fallback, c as escape_html, b as attr_class, d as slot, ad as bind_props, a8 as stringify } from "./index2.js";
const SLEEP = {
  /** Trace count that triggers auto-sleep */
  traceThreshold: 50
};
function MetricCard($$renderer, $$props) {
  let title = $$props["title"];
  let value = fallback($$props["value"], "—");
  let subtitle = fallback($$props["subtitle"], "");
  let status = fallback($$props["status"], "none");
  $$renderer.push(`<div class="metric-card svelte-1e9wj76"><div class="metric-header svelte-1e9wj76"><span class="metric-title svelte-1e9wj76">${escape_html(title)}</span> `);
  if (status !== "none") {
    $$renderer.push("<!--[0-->");
    $$renderer.push(`<span${attr_class(`badge badge-${stringify(status)}`, "svelte-1e9wj76")}>${escape_html(status === "ok" ? "healthy" : status === "warn" ? "warning" : "error")}</span>`);
  } else {
    $$renderer.push("<!--[-1-->");
  }
  $$renderer.push(`<!--]--></div> <div class="metric-value svelte-1e9wj76">${escape_html(value)}</div> `);
  if (subtitle) {
    $$renderer.push("<!--[0-->");
    $$renderer.push(`<div class="metric-subtitle svelte-1e9wj76">${escape_html(subtitle)}</div>`);
  } else {
    $$renderer.push("<!--[-1-->");
  }
  $$renderer.push(`<!--]--> <!--[-->`);
  slot($$renderer, $$props, "default", {});
  $$renderer.push(`<!--]--></div>`);
  bind_props($$props, { title, value, subtitle, status });
}
export {
  MetricCard as M,
  SLEEP as S
};
