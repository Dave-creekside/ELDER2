import { a9 as head, e as ensure_array_like, c as escape_html, b as attr_class, a8 as stringify } from "../../../chunks/index2.js";
import { o as onDestroy } from "../../../chunks/index-server.js";
import "../../../chunks/graph.js";
import "d3";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let seedNode = "";
    let dimension = 0;
    let rSquared = 0;
    let nodeNames = [];
    let radiusStats = [];
    onDestroy(() => {
    });
    head("47p7xq", $$renderer2, ($$renderer3) => {
      $$renderer3.title(($$renderer4) => {
        $$renderer4.push(`<title>ELDER2 — Fractal</title>`);
      });
    });
    $$renderer2.push(`<div class="page svelte-47p7xq"><div class="graph-area svelte-47p7xq"><div class="scale-panel svelte-47p7xq"><div class="scale-label svelte-47p7xq">r = 1</div></div> <div class="scale-panel svelte-47p7xq"><div class="scale-label svelte-47p7xq">r = 2</div></div> <div class="scale-panel svelte-47p7xq"><div class="scale-label svelte-47p7xq">r = 3</div></div> <div class="scale-panel svelte-47p7xq"><div class="scale-label svelte-47p7xq">Full graph</div></div></div> <aside class="stats-panel svelte-47p7xq"><label class="seed-select svelte-47p7xq"><span class="seed-label svelte-47p7xq">Seed Node</span> `);
    $$renderer2.select(
      { value: seedNode, class: "" },
      ($$renderer3) => {
        $$renderer3.push(`<!--[-->`);
        const each_array = ensure_array_like(nodeNames);
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let name = each_array[$$index];
          $$renderer3.option({ value: name }, ($$renderer4) => {
            $$renderer4.push(`${escape_html(name)}`);
          });
        }
        $$renderer3.push(`<!--]-->`);
      },
      "svelte-47p7xq"
    );
    $$renderer2.push(`</label> <div class="stat-hero svelte-47p7xq"><div class="stat-block svelte-47p7xq"><span class="stat-key svelte-47p7xq">D</span> <span class="stat-val svelte-47p7xq">${escape_html(dimension.toFixed(3))}</span></div> <div class="stat-block svelte-47p7xq"><span class="stat-key svelte-47p7xq">R²</span> <span class="stat-val svelte-47p7xq">${escape_html(rSquared.toFixed(3))}</span></div></div> <div class="divider svelte-47p7xq"></div> <div class="radius-list svelte-47p7xq"><div class="radius-heading svelte-47p7xq">Per-radius</div> <!--[-->`);
    const each_array_1 = ensure_array_like(radiusStats);
    for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
      let rs = each_array_1[$$index_1];
      $$renderer2.push(`<div class="radius-row svelte-47p7xq"><span class="radius-label svelte-47p7xq">${escape_html(rs.radius === "full" ? "Full" : `r = ${rs.radius}`)}</span> <div class="radius-detail svelte-47p7xq"><span>${escape_html(rs.nodeCount)} nodes</span> <span>${escape_html(rs.edgeCount)} edges</span></div> `);
      if (rs.radius !== "full") {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="radius-log svelte-47p7xq">ln(r) = ${escape_html(rs.logR.toFixed(2))}, ln(n) = ${escape_html(rs.logN.toFixed(2))}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    }
    $$renderer2.push(`<!--]--></div> <div class="divider svelte-47p7xq"></div> <div class="radius-list svelte-47p7xq"><div class="radius-heading svelte-47p7xq">Dimensionality</div> <div class="dim-row svelte-47p7xq"><span class="dim-label svelte-47p7xq">Hausdorff (fractal)</span> <span class="dim-val svelte-47p7xq">${escape_html(dimension.toFixed(4))}</span></div> <div class="dim-row svelte-47p7xq"><span class="dim-label svelte-47p7xq">Topological</span> <span class="dim-val svelte-47p7xq">${escape_html("—")}</span></div> <div class="dim-row svelte-47p7xq"><span class="dim-label svelte-47p7xq">Growth rate</span> <span class="dim-val svelte-47p7xq">${escape_html("—")}</span></div> <div class="dim-row svelte-47p7xq"><span class="dim-label svelte-47p7xq">Fit quality</span> <span${attr_class(`dim-val ${stringify("poor")}`, "svelte-47p7xq")}>${escape_html("Weak")}</span></div></div></aside></div>`);
  });
}
export {
  _page as default
};
