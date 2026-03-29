import { d as derived, w as writable } from "./index.js";
import "./socket.js";
const messages = writable([]);
const isThinking = writable(false);
derived(messages, ($m) => $m.length);
export {
  isThinking as i,
  messages as m
};
