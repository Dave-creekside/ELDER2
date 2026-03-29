import { a9 as head, c as escape_html, s as store_get, u as unsubscribe_stores } from "../../../chunks/index2.js";
import { o as onDestroy } from "../../../chunks/index-server.js";
import { g as graphData, n as nodeCount, e as edgeCount } from "../../../chunks/graph.js";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    function render(data) {
      return;
    }
    const unsubscribe = graphData.subscribe((data) => render());
    onDestroy(unsubscribe);
    head("3v9kyr", $$renderer2, ($$renderer3) => {
      $$renderer3.title(($$renderer4) => {
        $$renderer4.push(`<title>ELDER2 — Heatmap</title>`);
      });
    });
    $$renderer2.push(`<div class="page svelte-3v9kyr"><div class="header svelte-3v9kyr"><h1 class="svelte-3v9kyr">Dimensional Heatmap</h1> <span class="subtitle svelte-3v9kyr">${escape_html(store_get($$store_subs ??= {}, "$nodeCount", nodeCount))} nodes · ${escape_html(store_get($$store_subs ??= {}, "$edgeCount", edgeCount))} edges</span> `);
    {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div> <canvas class="svelte-3v9kyr"></canvas></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _page as default
};
