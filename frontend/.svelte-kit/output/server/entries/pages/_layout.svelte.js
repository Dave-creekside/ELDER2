import { g as getContext, e as ensure_array_like, a as attr, b as attr_class, s as store_get, c as escape_html, d as slot, u as unsubscribe_stores } from "../../chunks/index2.js";
import "clsx";
import "@sveltejs/kit/internal";
import "../../chunks/exports.js";
import "../../chunks/utils.js";
import "@sveltejs/kit/internal/server";
import "../../chunks/root.js";
import "../../chunks/state.svelte.js";
import { c as connectionStatus } from "../../chunks/socket.js";
import { n as nodeCount, e as edgeCount } from "../../chunks/graph.js";
import "../../chunks/chat.js";
const getStores = () => {
  const stores$1 = getContext("__svelte__");
  return {
    /** @type {typeof page} */
    page: {
      subscribe: stores$1.page.subscribe
    },
    /** @type {typeof navigating} */
    navigating: {
      subscribe: stores$1.navigating.subscribe
    },
    /** @type {typeof updated} */
    updated: stores$1.updated
  };
};
const page = {
  subscribe(fn) {
    const store = getStores().page;
    return store.subscribe(fn);
  }
};
function _layout($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    const NAV = [
      { href: "/", icon: "💬", label: "Chat" },
      { href: "/graph", icon: "🕸️", label: "Graph" },
      { href: "/galaxy", icon: "🌌", label: "Galaxy" },
      { href: "/heatmap", icon: "🔥", label: "Heatmap" },
      { href: "/matrix", icon: "📊", label: "Matrix" },
      { href: "/fractal", icon: "🔬", label: "Fractal" },
      { href: "/health", icon: "🩺", label: "Health" },
      { href: "/student", icon: "🎓", label: "Student" }
    ];
    $$renderer2.push(`<div class="shell svelte-12qhfyh"><nav class="sidebar svelte-12qhfyh"><div class="sidebar-brand svelte-12qhfyh"><span class="brand-icon svelte-12qhfyh">🧠</span> <span class="brand-text svelte-12qhfyh">ELDER<span class="brand-ver svelte-12qhfyh">2</span></span></div> <div class="sidebar-nav svelte-12qhfyh"><!--[-->`);
    const each_array = ensure_array_like(NAV);
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let item = each_array[$$index];
      $$renderer2.push(`<a${attr("href", item.href)}${attr_class("nav-link svelte-12qhfyh", void 0, {
        "active": store_get($$store_subs ??= {}, "$page", page).url.pathname === item.href
      })}><span class="nav-icon svelte-12qhfyh">${escape_html(item.icon)}</span> <span class="nav-label svelte-12qhfyh">${escape_html(item.label)}</span></a>`);
    }
    $$renderer2.push(`<!--]--></div> <div class="sidebar-footer svelte-12qhfyh"><div class="stat-row svelte-12qhfyh"><span class="stat-label svelte-12qhfyh">Nodes</span> <span class="stat-value svelte-12qhfyh">${escape_html(store_get($$store_subs ??= {}, "$nodeCount", nodeCount))}</span></div> <div class="stat-row svelte-12qhfyh"><span class="stat-label svelte-12qhfyh">Edges</span> <span class="stat-value svelte-12qhfyh">${escape_html(store_get($$store_subs ??= {}, "$edgeCount", edgeCount))}</span></div> <div class="status-row svelte-12qhfyh"><span${attr_class("status-dot svelte-12qhfyh", void 0, {
      "connected": store_get($$store_subs ??= {}, "$connectionStatus", connectionStatus) === "connected",
      "connecting": store_get($$store_subs ??= {}, "$connectionStatus", connectionStatus) === "connecting"
    })}></span> <span class="status-text svelte-12qhfyh">${escape_html(store_get($$store_subs ??= {}, "$connectionStatus", connectionStatus))}</span></div></div></nav> <main class="content svelte-12qhfyh"><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></main></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
export {
  _layout as default
};
