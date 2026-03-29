import { s as store_get, e as ensure_array_like, b as attr_class, a8 as stringify, c as escape_html, a as attr, u as unsubscribe_stores, a9 as head } from "../../chunks/index2.js";
import { m as messages, i as isThinking } from "../../chunks/chat.js";
function ChatTerminal($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let input = "";
    function formatTime(ts) {
      return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    $$renderer2.push(`<div class="chat svelte-q359by"><div class="messages svelte-q359by">`);
    if (store_get($$store_subs ??= {}, "$messages", messages).length === 0) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="empty-state svelte-q359by"><span class="empty-icon svelte-q359by">🧠</span> <p class="svelte-q359by">Begin a conversation with Elder</p></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> <!--[-->`);
    const each_array = ensure_array_like(store_get($$store_subs ??= {}, "$messages", messages));
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let msg = each_array[$$index];
      $$renderer2.push(`<div${attr_class(`message ${stringify(msg.role)}`, "svelte-q359by")}><div class="msg-header svelte-q359by"><span class="msg-role svelte-q359by">${escape_html(msg.role === "elder" ? "Elder" : msg.role === "user" ? "You" : "System")}</span> <span class="msg-time svelte-q359by">${escape_html(formatTime(msg.timestamp))}</span></div> <div class="msg-body svelte-q359by">${escape_html(msg.content)}</div></div>`);
    }
    $$renderer2.push(`<!--]--> `);
    if (store_get($$store_subs ??= {}, "$isThinking", isThinking)) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="message elder thinking svelte-q359by"><div class="msg-header svelte-q359by"><span class="msg-role svelte-q359by">Elder</span></div> <div class="msg-body svelte-q359by"><span class="thinking-dots svelte-q359by"><span class="svelte-q359by"></span><span class="svelte-q359by"></span><span class="svelte-q359by"></span></span></div></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div> <form class="input-bar svelte-q359by"><textarea placeholder="Message Elder..." rows="1" class="svelte-q359by">`);
    const $$body = escape_html(input);
    if ($$body) {
      $$renderer2.push(`${$$body}`);
    }
    $$renderer2.push(`</textarea> <button type="submit"${attr("disabled", !input.trim() || store_get($$store_subs ??= {}, "$isThinking", isThinking), true)} aria-label="Send message" class="svelte-q359by"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="svelte-q359by"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" class="svelte-q359by"></path></svg></button></form></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
function _page($$renderer) {
  head("1uha8ag", $$renderer, ($$renderer2) => {
    $$renderer2.title(($$renderer3) => {
      $$renderer3.push(`<title>ELDER2 — Chat</title>`);
    });
  });
  $$renderer.push(`<div class="page svelte-1uha8ag">`);
  ChatTerminal($$renderer);
  $$renderer.push(`<!----></div>`);
}
export {
  _page as default
};
