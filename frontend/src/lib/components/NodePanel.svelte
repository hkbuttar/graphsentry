<script lang="ts">
	/**
	 * Single-node detail panel, driven by the shared `selectedNodeId` store --
	 * clicking a node in NetworkGraph updates that store, which this
	 * component (an unrelated part of the tree) reacts to independently.
	 * This is the payoff of using a store instead of local component state:
	 * no props need to be threaded from NetworkGraph down to this sibling.
	 */
	import { getNode, type NodeDetail } from '$lib/api';
	import { selectedNodeId } from '$lib/stores';

	let node = $state<NodeDetail | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);

	$effect(() => {
		const id = $selectedNodeId;
		if (!id) {
			node = null;
			return;
		}
		loading = true;
		error = null;
		getNode(id)
			.then((data) => {
				node = data;
			})
			.catch((e) => {
				error = e instanceof Error ? e.message : String(e);
			})
			.finally(() => {
				loading = false;
			});
	});

	function labelText(label: number): string {
		if (label === 1) return 'illicit';
		if (label === 0) return 'licit';
		return 'unknown';
	}

	// a handful of the most legible raw features, rather than dumping all 165
	const previewFeatureKeys = ['time_step', 'pagerank', 'in_degree', 'out_degree', 'clustering', 'betweenness'];

	// Computed up front (and filtered to real numbers only) rather than
	// accessed ad-hoc in the template -- a `key in node.features` guard
	// directly inside the each-block was not a reliable guarantee that the
	// looked-up value is actually a number by the time `.toFixed()` runs
	// (caught by manual browser testing: an each-block nested inside a
	// conditional branch could observe a stale/transitional value during a
	// store update). Filtering here, once, before the template ever sees it,
	// removes that whole class of problem instead of guarding around it.
	const featureRows = $derived(
		node
			? previewFeatureKeys
					.filter((key) => typeof node!.features[key] === 'number')
					.map((key) => ({ key, value: node!.features[key] }))
			: []
	);
</script>

<div class="rounded-lg border border-slate-200 bg-white p-4">
	<h2 class="mb-3 text-sm font-semibold text-slate-900">Node lookup</h2>

	{#if !$selectedNodeId}
		<p class="text-sm text-slate-500">Click a node in the network view to inspect it.</p>
	{:else if loading}
		<p class="text-sm text-slate-500">Loading node {$selectedNodeId}…</p>
	{:else if error}
		<p class="text-sm text-red-600">{error}</p>
	{:else if node}
		<dl class="space-y-1 text-sm">
			<div class="flex justify-between"><dt class="text-slate-500">ID</dt><dd class="font-mono text-slate-900">{node.id}</dd></div>
			<div class="flex justify-between"><dt class="text-slate-500">Time step</dt><dd>{node.time_step}</dd></div>
			<div class="flex justify-between"><dt class="text-slate-500">Ground truth</dt><dd>{labelText(node.label)}</dd></div>
			<div class="flex justify-between">
				<dt class="text-slate-500">XGBoost P(illicit)</dt>
				<dd>{node.xgboost_proba !== null ? node.xgboost_proba.toFixed(4) : '—'}</dd>
			</div>
			<div class="flex justify-between">
				<dt class="text-slate-500">GraphSAGE P(illicit)</dt>
				<dd>{node.gnn_proba !== null ? node.gnn_proba.toFixed(4) : '—'}</dd>
			</div>
		</dl>

		<h3 class="mt-4 mb-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">Structural features</h3>
		<dl class="space-y-1 text-sm">
			{#each featureRows as row (row.key)}
				<div class="flex justify-between">
					<dt class="text-slate-500">{row.key}</dt>
					<dd class="tabular-nums">{row.value.toFixed(4)}</dd>
				</div>
			{/each}
		</dl>

		<h3 class="mt-4 mb-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">
			Neighbors ({node.in_neighbors.length} in / {node.out_neighbors.length} out)
		</h3>
		<div class="flex gap-4 text-xs">
			<div class="flex-1">
				<p class="mb-1 text-slate-500">Paid this node</p>
				<ul class="max-h-24 space-y-0.5 overflow-y-auto font-mono text-slate-700">
					{#each node.in_neighbors.slice(0, 20) as n (n)}
						<li>
							<button class="hover:underline" onclick={() => selectedNodeId.set(n)}>{n}</button>
						</li>
					{/each}
				</ul>
			</div>
			<div class="flex-1">
				<p class="mb-1 text-slate-500">Paid by this node</p>
				<ul class="max-h-24 space-y-0.5 overflow-y-auto font-mono text-slate-700">
					{#each node.out_neighbors.slice(0, 20) as n (n)}
						<li>
							<button class="hover:underline" onclick={() => selectedNodeId.set(n)}>{n}</button>
						</li>
					{/each}
				</ul>
			</div>
		</div>
	{/if}
</div>
