import { d as derived, w as writable } from "./index.js";
import "socket.io-client";
const connectionStatus = writable("disconnected");
derived(connectionStatus, ($s) => $s === "connected");
export {
  connectionStatus as c
};
