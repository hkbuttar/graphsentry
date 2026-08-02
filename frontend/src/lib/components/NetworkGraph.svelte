<script lang="ts">
	/**
	 * Force-directed network visualization. D3 owns the physics (node
	 * positions via a force simulation); Svelte owns the DOM (nodes/edges are
	 * rendered through normal Svelte template blocks, driven by plain,
	 * non-reactive arrays that the simulation mutates directly).
	 *
	 * Deliberately NOT feeding Svelte's `$state`-proxied arrays into
	 * `forceSimulation` -- d3-force expects to freely mutate plain objects in
	 * place every tick, and running that through Svelte 5's reactive proxy
	 * wrapper is an easy source of subtle bugs. Instead, the simulation runs
	 * on plain arrays, and each tick copies the current positions into a
	 * separate `$state` array that the template actually reads.
	 */
	import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide, zoom, select, type Simulation } from 'd3';
	import { getNetwork, type NetworkNode } from '$lib/api';
	import { selectedNodeId, activeModel, probabilityThreshold, activeTimeStep } from '$lib/stores';

	const width = 760;
	const height = 560;

	type SimNode = NetworkNode & { x?: number; y?: number; vx?: number; vy?: number };
	type RenderLink = { sx: number; sy: number; tx: number; ty: number };

	let svgEl: SVGSVGElement | undefined = $state();
	let gEl: SVGGElement | undefined = $state();

	let renderNodes = $state<SimNode[]>([]);
	let renderLinks = $state<RenderLink[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let truncated = $state(false);

	let simNodes: SimNode[] = [];
	let simLinks: { source: string | SimNode; target: string | SimNode }[] = [];
	let simulation: Simulation<SimNode, undefined> | null = null;

	function endpointX(e: string | SimNode): number {
		return typeof e === 'object' ? (e.x ?? 0) : 0;
	}
	function endpointY(e: string | SimNode): number {
		return typeof e === 'object' ? (e.y ?? 0) : 0;
	}

	function syncRenderState() {
		renderNodes = simNodes.map((n) => ({ ...n }));
		renderLinks = simLinks.map((l) => ({
			sx: endpointX(l.source),
			sy: endpointY(l.source),
			tx: endpointX(l.target),
			ty: endpointY(l.target)
		}));
	}

	async function loadNetwork(timeStep: number) {
		loading = true;
		error = null;
		try {
			const data = await getNetwork(timeStep);
			truncated = data.truncated;
			simNodes = data.nodes.map((n) => ({ ...n }));
			simLinks = data.edges.map((e) => ({ source: e.source, target: e.target }));

			simulation?.stop();
			simulation = forceSimulation(simNodes)
				.force(
					'link',
					forceLink(simLinks)
						.id((d) => (d as SimNode).id)
						.distance(24)
						.strength(0.3)
				)
				.force('charge', forceManyBody().strength(-40))
				.force('center', forceCenter(width / 2, height / 2))
				.force('collide', forceCollide(7))
				.on('tick', syncRenderState);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		loadNetwork($activeTimeStep);
	});

	function proba(node: NetworkNode): number | null {
		return $activeModel === 'xgboost' ? node.xgboost_proba : node.gnn_proba;
	}

	function fillColor(node: NetworkNode): string {
		const p = proba(node);
		if (p === null) return '#cbd5e1'; // slate-300 -- no prediction (unlabeled node)
		return p >= $probabilityThreshold ? '#dc2626' : '#2563eb'; // red-600 : blue-600
	}

	// zoom/pan
	$effect(() => {
		if (!svgEl || !gEl) return;
		const behavior = zoom<SVGSVGElement, unknown>()
			.scaleExtent([0.2, 5])
			.on('zoom', (event) => {
				select(gEl!).attr('transform', event.transform.toString());
			});
		select(svgEl).call(behavior);
		return () => {
			select(svgEl!).on('.zoom', null);
		};
	});
</script>

<div class="relative">
	{#if loading}
		<div class="absolute inset-0 z-10 flex items-center justify-center text-sm text-slate-500">
			Loading network…
		</div>
	{/if}
	{#if error}
		<div class="absolute inset-0 z-10 flex items-center justify-center text-sm text-red-600">
			{error}
		</div>
	{/if}

	<svg
		bind:this={svgEl}
		width="100%"
		height={height}
		viewBox="0 0 {width} {height}"
		class="rounded-lg border border-slate-200 bg-white"
	>
		<g bind:this={gEl}>
			{#each renderLinks as link, i (i)}
				<line x1={link.sx} y1={link.sy} x2={link.tx} y2={link.ty} stroke="#e2e8f0" stroke-width="1" />
			{/each}
			{#each renderNodes as node (node.id)}
				<circle
					cx={node.x}
					cy={node.y}
					r={node.id === $selectedNodeId ? 8 : 5}
					fill={fillColor(node)}
					stroke={node.label === 1 ? '#0f172a' : node.id === $selectedNodeId ? '#f59e0b' : 'none'}
					stroke-width={node.label === 1 ? 2 : node.id === $selectedNodeId ? 3 : 0}
					class="cursor-pointer"
					role="button"
					tabindex="0"
					onclick={() => selectedNodeId.set(node.id)}
					onkeydown={(e) => e.key === 'Enter' && selectedNodeId.set(node.id)}
				>
					<title>{node.id}</title>
				</circle>
			{/each}
		</g>
	</svg>

	{#if truncated}
		<p class="mt-1 text-xs text-slate-500">
			Showing a subsample -- this time step's full component is larger than the render cap.
		</p>
	{/if}

	<div class="mt-2 flex flex-wrap gap-4 text-xs text-slate-600">
		<span class="flex items-center gap-1"><span class="inline-block h-3 w-3 rounded-full bg-red-600"></span> predicted illicit</span>
		<span class="flex items-center gap-1"><span class="inline-block h-3 w-3 rounded-full bg-blue-600"></span> predicted licit</span>
		<span class="flex items-center gap-1"><span class="inline-block h-3 w-3 rounded-full bg-slate-300"></span> no prediction (unlabeled)</span>
		<span class="flex items-center gap-1"><span class="inline-block h-3 w-3 rounded-full border-2 border-slate-900 bg-white"></span> actually illicit (ground truth)</span>
	</div>
</div>
