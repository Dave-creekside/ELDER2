export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.CBJ7VIrV.js",app:"_app/immutable/entry/app.BH1DOsFx.js",imports:["_app/immutable/entry/start.CBJ7VIrV.js","_app/immutable/chunks/BK2liVwn.js","_app/immutable/chunks/d7fLZnZ8.js","_app/immutable/entry/app.BH1DOsFx.js","_app/immutable/chunks/d7fLZnZ8.js","_app/immutable/chunks/D9j3YgkD.js","_app/immutable/chunks/BySx5bR9.js","_app/immutable/chunks/DtG7a6iu.js","_app/immutable/chunks/B_SXgKcy.js","_app/immutable/chunks/B0-H3yY_.js","_app/immutable/chunks/BLyE5bgV.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js')),
			__memo(() => import('./nodes/3.js')),
			__memo(() => import('./nodes/4.js')),
			__memo(() => import('./nodes/5.js')),
			__memo(() => import('./nodes/6.js')),
			__memo(() => import('./nodes/7.js')),
			__memo(() => import('./nodes/8.js')),
			__memo(() => import('./nodes/9.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			},
			{
				id: "/fractal",
				pattern: /^\/fractal\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 3 },
				endpoint: null
			},
			{
				id: "/galaxy",
				pattern: /^\/galaxy\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 4 },
				endpoint: null
			},
			{
				id: "/graph",
				pattern: /^\/graph\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 5 },
				endpoint: null
			},
			{
				id: "/health",
				pattern: /^\/health\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 6 },
				endpoint: null
			},
			{
				id: "/heatmap",
				pattern: /^\/heatmap\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 7 },
				endpoint: null
			},
			{
				id: "/matrix",
				pattern: /^\/matrix\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 8 },
				endpoint: null
			},
			{
				id: "/student",
				pattern: /^\/student\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 9 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
