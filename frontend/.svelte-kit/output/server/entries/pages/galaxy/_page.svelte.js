import { a9 as head } from "../../../chunks/index2.js";
import { o as onDestroy } from "../../../chunks/index-server.js";
import "../../../chunks/graph.js";
import * as THREE from "three";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let animId;
    new THREE.Raycaster();
    new THREE.Vector2();
    new THREE.Plane();
    new THREE.Vector3();
    new THREE.Vector3();
    function handleResize() {
      return;
    }
    onDestroy(() => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
    });
    head("28uhd9", $$renderer2, ($$renderer3) => {
      $$renderer3.title(($$renderer4) => {
        $$renderer4.push(`<title>ELDER2 — Galaxy</title>`);
      });
    });
    $$renderer2.push(`<div class="page svelte-28uhd9"></div>`);
  });
}
export {
  _page as default
};
