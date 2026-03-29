import { a9 as head, c as escape_html, s as store_get, u as unsubscribe_stores } from "../../../chunks/index2.js";
import { o as onDestroy } from "../../../chunks/index-server.js";
import { g as graphData, n as nodeCount } from "../../../chunks/graph.js";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let stats = { diameter: 0, density: 0 };
    function render(data) {
      return;
    }
    const unsubscribe = graphData.subscribe((data) => render());
    onDestroy(unsubscribe);
    head("lqcok6", $$renderer2, ($$renderer3) => {
      $$renderer3.title(($$renderer4) => {
        $$renderer4.push(`<title>ELDER2 — Distance Matrix</title>`);
      });
    });
    $$renderer2.push(`<div class="page svelte-lqcok6"><div class="header svelte-lqcok6"><h1 class="svelte-lqcok6">Distance Matrix</h1> <span class="stat svelte-lqcok6">Diameter: ${escape_html(stats.diameter)}</span> <span class="stat svelte-lqcok6">Density: ${escape_html((stats.density * 100).toFixed(1))}%</span> <span class="subtitle svelte-lqcok6">${escape_html(store_get($$store_subs ??= {}, "$nodeCount", nodeCount))} nodes</span></div> <canvas class="svelte-lqcok6"></canvas></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
