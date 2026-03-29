

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export const universal = {
  "ssr": false,
  "prerender": false
};
export const universal_id = "src/routes/+layout.ts";
export const imports = ["_app/immutable/nodes/0.hK-NuZ7T.js","_app/immutable/chunks/BySx5bR9.js","_app/immutable/chunks/d7fLZnZ8.js","_app/immutable/chunks/8eKNP362.js","_app/immutable/chunks/BLyE5bgV.js","_app/immutable/chunks/D9j3YgkD.js","_app/immutable/chunks/CaSnGP60.js","_app/immutable/chunks/B_P2NTkB.js","_app/immutable/chunks/ePWG-NTR.js","_app/immutable/chunks/C4D3PsxB.js","_app/immutable/chunks/DNLdPdKK.js","_app/immutable/chunks/BK2liVwn.js","_app/immutable/chunks/DiS0JaPU.js","_app/immutable/chunks/BqWHK34f.js"];
export const stylesheets = ["_app/immutable/assets/0.AwFfuDTB.css"];
export const fonts = [];
