<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphData, type GraphNode, type GraphEdge, type Hyperedge } from '$lib/stores/graph';
  import { CORE_NODES } from '$lib/config';
  import { NODE_COLORS, CORE_NODE_COLOR } from '$lib/utils/colors';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
  import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
  import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

  function hexToThree(hex: string): number {
    return parseInt(hex.replace('#', ''), 16);
  }

  function hashId(s: string): number {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = s.charCodeAt(i) + ((h << 5) - h);
    return Math.abs(h);
  }

  // Even distribution on a sphere via fibonacci spiral
  function fibSphere(index: number, total: number, radius: number): [number, number, number] {
    if (total <= 1) return [0, 0, 0];
    const golden = Math.PI * (3 - Math.sqrt(5));
    const y = 1 - (index / (total - 1)) * 2;
    const r = Math.sqrt(1 - y * y) * radius;
    const theta = golden * index;
    return [Math.cos(theta) * r, y * radius, Math.sin(theta) * r];
  }

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let composer: EffectComposer;
  let animId: number;
  let clock = 0;
  let lastNodeCount = -1;

  let nodeGroup: THREE.Group;
  let edgeGroup: THREE.Group;
  let labelGroup: THREE.Group;
  let hyperedgeGroup: THREE.Group;

  // Interaction state
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const dragPlane = new THREE.Plane();
  const intersection = new THREE.Vector3();
  let draggedMesh: THREE.Mesh | null = null;
  let dragOffset = new THREE.Vector3();
  let hoveredNode: string | null = null;

  // Scene data
  interface SceneNode {
    id: string;
    label: string;
    isCore: boolean;
    color: number;
    size: number;
    distance: number;
    mesh: THREE.Mesh;
    sprite: THREE.Sprite;
    glowRing: THREE.Mesh;
    connectedIds: Set<string>;
  }

  interface SceneEdge {
    source: string;
    target: string;
    strength: number;
    line: THREE.Line;
    baseColor: number;
  }

  let sceneNodes: SceneNode[] = [];
  let sceneEdges: SceneEdge[] = [];
  let nodeById = new Map<string, SceneNode>();

  // ── Setup ──────────────────────────────────────────────────────

  function init() {
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x030508, 0.0012);

    camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 4000);
    camera.position.set(0, 140, 320);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 60;
    controls.maxDistance = 600;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.4;

    // Lighting — subtle, lets bloom/emissive do the work
    scene.add(new THREE.AmbientLight(0x0a0e18, 0.4));
    const coreLight = new THREE.PointLight(hexToThree(CORE_NODE_COLOR), 0.5, 500);
    coreLight.position.set(0, 40, 0);
    scene.add(coreLight);
    const fillLight = new THREE.PointLight(0x0055aa, 0.25, 400);
    fillLight.position.set(-80, -40, 80);
    scene.add(fillLight);

    nodeGroup = new THREE.Group();
    edgeGroup = new THREE.Group();
    labelGroup = new THREE.Group();
    hyperedgeGroup = new THREE.Group();
    scene.add(edgeGroup);
    scene.add(hyperedgeGroup);
    scene.add(nodeGroup);
    scene.add(labelGroup);

    createStarfield();

    // Post-processing — lower bloom than before to avoid washout
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 1.6, 0.5, 0.35));

    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
  }

  function createStarfield() {
    const count = 2500;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 400 + Math.random() * 1400;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
      color: 0xffffff, size: 1, transparent: true, opacity: 0.35, sizeAttenuation: true,
    })));
  }

  // ── Label ──────────────────────────────────────────────────────

  function makeLabel(text: string, isCore: boolean): THREE.Sprite {
    const c = document.createElement('canvas');
    const ctx = c.getContext('2d')!;
    const fs = isCore ? 48 : 36;
    const weight = isCore ? '600' : '400';
    ctx.font = `${weight} ${fs}px sans-serif`;
    const tw = ctx.measureText(text).width;
    c.width = tw + 24;
    c.height = fs + 20;
    ctx.font = `${weight} ${fs}px sans-serif`;
    ctx.shadowColor = 'rgba(0,0,0,0.9)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, c.width / 2, c.height / 2);
    const tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.9, depthWrite: false });
    const s = new THREE.Sprite(mat);
    s.scale.set(c.width / 7, c.height / 7, 1);
    return s;
  }

  // ── Build ──────────────────────────────────────────────────────

  function buildScene(data: { nodes: GraphNode[]; edges: GraphEdge[]; hyperedges: Hyperedge[] }) {
    if (!scene || !data.nodes.length) return;

    // Tear down previous objects
    [nodeGroup, edgeGroup, labelGroup, hyperedgeGroup].forEach(g => {
      while (g.children.length) {
        const child = g.children[0];
        g.remove(child);
        if ((child as any).geometry) (child as any).geometry.dispose();
        if ((child as any).material) {
          const m = (child as any).material;
          if (Array.isArray(m)) m.forEach((x: any) => x.dispose());
          else m.dispose();
        }
      }
    });

    sceneNodes = [];
    sceneEdges = [];
    nodeById = new Map();

    // ── BFS from core nodes (same algorithm as graph page) ──

    const adj = new Map<string, Set<string>>();
    data.edges.forEach(e => {
      if (!adj.has(e.source)) adj.set(e.source, new Set());
      if (!adj.has(e.target)) adj.set(e.target, new Set());
      adj.get(e.source)!.add(e.target);
      adj.get(e.target)!.add(e.source);
    });

    const coreIds = data.nodes
      .filter(n => CORE_NODES.includes(n.label as any))
      .map(n => n.id);

    const distances = new Map<string, number>();
    const queue = [...coreIds];
    coreIds.forEach(id => distances.set(id, 0));
    while (queue.length) {
      const cur = queue.shift()!;
      const d = distances.get(cur)!;
      for (const nb of adj.get(cur) ?? []) {
        if (!distances.has(nb)) {
          distances.set(nb, d + 1);
          queue.push(nb);
        }
      }
    }

    // Pre-compute connectivity for hover
    const connectedMap = new Map<string, Set<string>>();
    data.nodes.forEach(n => connectedMap.set(n.id, new Set()));
    data.edges.forEach(e => {
      connectedMap.get(e.source)?.add(e.target);
      connectedMap.get(e.target)?.add(e.source);
    });

    // Group nodes by distance ring
    const byDistance = new Map<number, GraphNode[]>();
    data.nodes.forEach(n => {
      const raw = distances.get(n.id) ?? 999;
      const d = raw === 999 ? 4 : raw;
      if (!byDistance.has(d)) byDistance.set(d, []);
      byDistance.get(d)!.push(n);
    });

    // Shell radii — core tight, then expanding shells
    const SHELL_R: Record<number, number> = { 0: 28, 1: 75, 2: 130, 3: 185, 4: 240 };

    // ── Place nodes on spherical shells ──

    byDistance.forEach((ring, dist) => {
      const radius = SHELL_R[Math.min(dist, 4)] ?? 240;

      ring.forEach((n, i) => {
        const isCore = CORE_NODES.includes(n.label as any);
        const importance = (n as any).importance ?? 5;
        const baseSize = isCore ? 5 : 2.5 + (importance / 10) * 3;
        const colorHex = isCore ? CORE_NODE_COLOR : NODE_COLORS[hashId(n.id) % NODE_COLORS.length];
        const color = hexToThree(colorHex);

        let x: number, y: number, z: number;
        if (dist === 0) {
          // Core nodes in a small ring at center
          const angle = (i * 2 * Math.PI) / Math.max(ring.length, 3) - Math.PI / 2;
          x = Math.cos(angle) * radius;
          y = Math.sin(i * 1.2) * 10;
          z = Math.sin(angle) * radius;
        } else {
          [x, y, z] = fibSphere(i, ring.length, radius);
        }

        // Main sphere — emissive for bloom pickup
        const geo = new THREE.SphereGeometry(baseSize, 24, 24);
        const mat = new THREE.MeshPhongMaterial({
          color, emissive: color, emissiveIntensity: 0.8,
          shininess: 200, transparent: true, opacity: 0.95,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, y, z);
        mesh.userData = { nodeId: n.id };

        // Inner white core (like graph page's inner highlight circle)
        mesh.add(new THREE.Mesh(
          new THREE.SphereGeometry(baseSize * 0.35, 12, 12),
          new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 }),
        ));

        // Glow ring — like graph page's outer glow stroke
        const glowGeo = new THREE.SphereGeometry(baseSize * (isCore ? 2.8 : 2.2), 16, 16);
        const glowMat = new THREE.MeshBasicMaterial({
          color, transparent: true, opacity: 0.12,
          side: THREE.BackSide, blending: THREE.AdditiveBlending,
        });
        const glowRing = new THREE.Mesh(glowGeo, glowMat);
        mesh.add(glowRing);

        nodeGroup.add(mesh);

        // Label
        const sprite = makeLabel(n.label, isCore);
        sprite.position.set(x, y + baseSize + 4, z);
        labelGroup.add(sprite);

        const sn: SceneNode = {
          id: n.id, label: n.label, isCore, color, size: baseSize,
          distance: dist, mesh, sprite, glowRing,
          connectedIds: connectedMap.get(n.id) ?? new Set(),
        };
        sceneNodes.push(sn);
        nodeById.set(n.id, sn);
      });
    });

    // ── Curved 3D edges (bezier arcs toward center) ──

    const nodeIdSet = new Set(sceneNodes.map(n => n.id));

    data.edges.forEach(e => {
      if (!nodeIdSet.has(e.source) || !nodeIdSet.has(e.target)) return;
      const src = nodeById.get(e.source)!;
      const tgt = nodeById.get(e.target)!;
      const strength = e.strength ?? 0.5;

      const points = curveEdge(src.mesh.position, tgt.mesh.position);

      let baseColor: number;
      if (strength > 0.7) baseColor = 0xffcc00;
      else if (strength < 0.3) baseColor = 0x4466ff;
      else baseColor = hexToThree(CORE_NODE_COLOR);

      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: baseColor, transparent: true,
        opacity: 0.1 + strength * 0.25,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      edgeGroup.add(line);

      sceneEdges.push({ source: e.source, target: e.target, strength, line, baseColor });
    });

    // ── Hyperedges as 3D closed catmull-rom curves ──

    if (data.hyperedges?.length) {
      data.hyperedges.filter(he => he.members.length >= 2).forEach(he => {
        const members = he.members.map(id => nodeById.get(id)).filter(Boolean) as SceneNode[];
        if (members.length < 2) return;

        const centroid = new THREE.Vector3();
        members.forEach(m => centroid.add(m.mesh.position));
        centroid.divideScalar(members.length);

        // Sort by angle around centroid (xz plane)
        const sorted = [...members].sort((a, b) => {
          const aa = Math.atan2(a.mesh.position.z - centroid.z, a.mesh.position.x - centroid.x);
          const ba = Math.atan2(b.mesh.position.z - centroid.z, b.mesh.position.x - centroid.x);
          return aa - ba;
        });

        const pts = sorted.map(m => m.mesh.position.clone());
        pts.push(pts[0].clone()); // close the loop

        if (pts.length >= 3) {
          const curve = new THREE.CatmullRomCurve3(pts, true);
          const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(pts.length * 12));
          const mat = new THREE.LineDashedMaterial({
            color: hexToThree(CORE_NODE_COLOR),
            transparent: true, opacity: 0.12,
            dashSize: 3, gapSize: 3,
          });
          const line = new THREE.Line(geo, mat);
          line.computeLineDistances();
          line.userData = { memberIds: he.members };
          hyperedgeGroup.add(line);
        }
      });
    }

    lastNodeCount = data.nodes.length;
  }

  /** Build a curved edge: quadratic bezier arcing inward toward origin */
  function curveEdge(a: THREE.Vector3, b: THREE.Vector3): THREE.Vector3[] {
    const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
    const dist = a.distanceTo(b);
    // Pull midpoint toward origin for inward arc
    const inward = mid.clone().normalize().multiplyScalar(-dist * 0.15);
    // Slight upward lift
    const lift = new THREE.Vector3(0, dist * 0.08, 0);
    const ctrl = mid.clone().add(inward).add(lift);
    return new THREE.QuadraticBezierCurve3(a.clone(), ctrl, b.clone()).getPoints(20);
  }

  // ── Interaction ────────────────────────────────────────────────

  function updateMouse(e: PointerEvent) {
    const r = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  }

  function onPointerMove(e: PointerEvent) {
    if (draggedMesh) {
      updateMouse(e);
      raycaster.setFromCamera(mouse, camera);
      if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
        const pos = intersection.add(dragOffset);
        draggedMesh.position.copy(pos);
        const sn = nodeById.get(draggedMesh.userData.nodeId);
        if (sn) {
          sn.sprite.position.set(pos.x, pos.y + sn.size + 4, pos.z);
          rebuildEdgesFor(sn.id);
        }
      }
      return;
    }

    // Hover
    updateMouse(e);
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(
      sceneNodes.map(n => n.mesh), false,
    );
    const newHover = hits.length ? (hits[0].object as THREE.Mesh).userData.nodeId : null;
    if (newHover !== hoveredNode) {
      hoveredNode = newHover;
      applyHover();
    }
  }

  function onPointerDown(e: PointerEvent) {
    updateMouse(e);
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(sceneNodes.map(n => n.mesh), false);
    if (hits.length) {
      draggedMesh = hits[0].object as THREE.Mesh;
      controls.enabled = false;
      controls.autoRotate = false;
      dragPlane.setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()).negate(),
        draggedMesh.position,
      );
      raycaster.ray.intersectPlane(dragPlane, intersection);
      dragOffset.subVectors(draggedMesh.position, intersection);
    }
  }

  function onPointerUp() {
    draggedMesh = null;
    controls.enabled = true;
    controls.autoRotate = true;
  }

  function rebuildEdgesFor(nodeId: string) {
    sceneEdges.forEach(se => {
      if (se.source !== nodeId && se.target !== nodeId) return;
      const src = nodeById.get(se.source);
      const tgt = nodeById.get(se.target);
      if (!src || !tgt) return;
      const points = curveEdge(src.mesh.position, tgt.mesh.position);
      se.line.geometry.dispose();
      se.line.geometry = new THREE.BufferGeometry().setFromPoints(points);
    });
  }

  /** Dim/brighten to match graph page hover behavior */
  function applyHover() {
    if (hoveredNode) {
      const hn = nodeById.get(hoveredNode);
      if (!hn) return;
      const connected = new Set(hn.connectedIds);
      connected.add(hoveredNode);

      sceneNodes.forEach(sn => {
        const hit = connected.has(sn.id);
        const mat = sn.mesh.material as THREE.MeshPhongMaterial;
        mat.opacity = hit ? 1 : 0.12;
        mat.emissiveIntensity = hit ? 1.4 : 0.15;
        (sn.glowRing.material as THREE.MeshBasicMaterial).opacity =
          sn.id === hoveredNode ? 0.45 : hit ? 0.2 : 0.01;
        (sn.sprite.material as THREE.SpriteMaterial).opacity = hit ? 1 : 0.08;
      });

      sceneEdges.forEach(se => {
        const mat = se.line.material as THREE.LineBasicMaterial;
        if (se.source === hoveredNode || se.target === hoveredNode) {
          mat.opacity = 0.7;
          mat.color.setHex(hexToThree(CORE_NODE_COLOR));
        } else {
          mat.opacity = 0.015;
        }
      });

      // Hyperedge highlight
      hyperedgeGroup.children.forEach(child => {
        const mat = (child as THREE.Line).material as THREE.LineDashedMaterial;
        const memberIds: string[] = child.userData.memberIds ?? [];
        mat.opacity = memberIds.includes(hoveredNode!) ? 0.5 : 0.02;
      });

    } else {
      // Restore defaults
      sceneNodes.forEach(sn => {
        const mat = sn.mesh.material as THREE.MeshPhongMaterial;
        mat.opacity = 0.95;
        mat.emissiveIntensity = 0.8;
        (sn.glowRing.material as THREE.MeshBasicMaterial).opacity = 0.12;
        (sn.sprite.material as THREE.SpriteMaterial).opacity = 0.9;
      });
      sceneEdges.forEach(se => {
        const mat = se.line.material as THREE.LineBasicMaterial;
        mat.opacity = 0.1 + se.strength * 0.25;
        mat.color.setHex(se.baseColor);
      });
      hyperedgeGroup.children.forEach(child => {
        ((child as THREE.Line).material as THREE.LineDashedMaterial).opacity = 0.12;
      });
    }
  }

  // ── Animate ────────────────────────────────────────────────────

  function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    clock += 0.016;

    if (!hoveredNode) {
      // Subtle edge pulse
      sceneEdges.forEach((se, i) => {
        const mat = se.line.material as THREE.LineBasicMaterial;
        const base = 0.1 + se.strength * 0.25;
        mat.opacity = base + Math.sin(clock * 1.2 + i * 0.5) * 0.05;
      });

      // Gentle glow breathing
      sceneNodes.forEach((sn, i) => {
        const mat = sn.glowRing.material as THREE.MeshBasicMaterial;
        mat.opacity = 0.12 + Math.sin(clock * 0.6 + i * 0.4) * 0.04;
      });
    }

    composer.render();
  }

  // ── Lifecycle ──────────────────────────────────────────────────

  function handleResize() {
    if (!container || !renderer) return;
    const r = container.getBoundingClientRect();
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
    renderer.setSize(r.width, r.height);
    composer.setSize(r.width, r.height);
  }

  let unsubscribe: () => void;

  onMount(() => {
    init();
    animate();
    window.addEventListener('resize', handleResize);
    unsubscribe = graphData.subscribe(data => {
      if (scene && data.nodes.length > 0 && data.nodes.length !== lastNodeCount) {
        buildScene(data);
      }
    });
  });

  onDestroy(() => {
    if (unsubscribe) unsubscribe();
    cancelAnimationFrame(animId);
    window.removeEventListener('resize', handleResize);
    renderer?.domElement.removeEventListener('pointermove', onPointerMove);
    renderer?.domElement.removeEventListener('pointerdown', onPointerDown);
    renderer?.domElement.removeEventListener('pointerup', onPointerUp);
    renderer?.dispose();
    controls?.dispose();
  });
</script>

<svelte:head><title>ELDER2 — Galaxy</title></svelte:head>

<div class="page" bind:this={container}></div>

<style>
  .page {
    height: 100%;
    position: relative;
    background:
      radial-gradient(ellipse at 20% 30%, rgba(0, 100, 150, 0.1) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 70%, rgba(100, 0, 150, 0.07) 0%, transparent 50%),
      radial-gradient(ellipse at 50% 50%, rgba(0, 50, 100, 0.12) 0%, transparent 70%),
      radial-gradient(ellipse at center, #0a0e18 0%, #020408 100%);
    overflow: hidden;
  }

  .page :global(canvas) {
    display: block;
  }
</style>
