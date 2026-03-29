
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	type MatcherParam<M> = M extends (param : string) => param is (infer U extends string) ? U : string;

	export interface AppTypes {
		RouteId(): "/" | "/fractal" | "/galaxy" | "/graph" | "/health" | "/heatmap" | "/matrix" | "/student";
		RouteParams(): {
			
		};
		LayoutParams(): {
			"/": Record<string, never>;
			"/fractal": Record<string, never>;
			"/galaxy": Record<string, never>;
			"/graph": Record<string, never>;
			"/health": Record<string, never>;
			"/heatmap": Record<string, never>;
			"/matrix": Record<string, never>;
			"/student": Record<string, never>
		};
		Pathname(): "/" | "/fractal" | "/galaxy" | "/graph" | "/health" | "/heatmap" | "/matrix" | "/student";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}