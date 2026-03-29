import { d as derived, w as writable } from "./index.js";
import "./socket.js";
const graphData = writable({ nodes: [], edges: [], hyperedges: [] });
const nodeCount = derived(graphData, ($g) => $g.nodes.length);
const edgeCount = derived(graphData, ($g) => $g.edges.length);
derived(graphData, ($g) => $g.hyperedges.length);
derived(graphData, ($g) => {
  if ($g.edges.length === 0) return 0;
  const sum = $g.edges.reduce((acc, e) => acc + (e.strength ?? 0), 0);
  return sum / $g.edges.length;
});
export {
  edgeCount as e,
  graphData as g,
  nodeCount as n
};
