<script lang="ts">
	/**
	 * Model comparison table -- the honest "does the GNN beat the baseline"
	 * result, surfaced visually instead of only living in the README.
	 */
	import { getMetrics, type ModelMetrics } from '$lib/api';

	let metrics = $state<ModelMetrics[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	$effect(() => {
		getMetrics()
			.then((data) => {
				metrics = data.metrics;
			})
			.catch((e) => {
				error = e instanceof Error ? e.message : String(e);
			})
			.finally(() => {
				loading = false;
			});
	});

	function fmt(n: number): string {
		return n.toFixed(4);
	}

	const testRows = $derived(metrics.filter((m) => m.split === 'test'));
	const trainRows = $derived(metrics.filter((m) => m.split === 'train'));
</script>

<div class="rounded-lg border border-slate-200 bg-white p-4">
	<h2 class="mb-1 text-sm font-semibold text-slate-900">Model comparison</h2>
	<p class="mb-3 text-xs text-slate-500">Test set = steps 35-49, the number that actually matters.</p>

	{#if loading}
		<p class="text-sm text-slate-500">Loading…</p>
	{:else if error}
		<p class="text-sm text-red-600">{error}</p>
	{:else}
		<table class="w-full text-sm">
			<thead>
				<tr class="border-b border-slate-200 text-left text-slate-500">
					<th class="py-1 pr-2 font-medium">Model</th>
					<th class="py-1 px-2 font-medium">Split</th>
					<th class="py-1 px-2 font-medium text-right">Precision</th>
					<th class="py-1 px-2 font-medium text-right">Recall</th>
					<th class="py-1 px-2 font-medium text-right">F1</th>
					<th class="py-1 pl-2 font-medium text-right">PR-AUC</th>
				</tr>
			</thead>
			<tbody>
				{#each [...testRows, ...trainRows] as row (row.model + row.split)}
					<tr class="border-b border-slate-100 last:border-0" class:font-semibold={row.split === 'test'}>
						<td class="py-1.5 pr-2 text-slate-900">{row.model}</td>
						<td class="py-1.5 px-2 text-slate-500">{row.split}</td>
						<td class="py-1.5 px-2 text-right tabular-nums">{fmt(row.precision)}</td>
						<td class="py-1.5 px-2 text-right tabular-nums">{fmt(row.recall)}</td>
						<td class="py-1.5 px-2 text-right tabular-nums">{fmt(row.f1)}</td>
						<td class="py-1.5 pl-2 text-right tabular-nums">{fmt(row.pr_auc)}</td>
					</tr>
				{/each}
			</tbody>
		</table>

		{#if testRows.length === 2}
			{@const xgb = testRows.find((r) => r.model === 'XGBoost')}
			{@const gnn = testRows.find((r) => r.model === 'GraphSAGE')}
			{#if xgb && gnn}
				<p class="mt-3 text-xs text-slate-500">
					GNN {gnn.f1 > xgb.f1 ? 'beats' : 'does not beat'} the baseline on test F1 ({fmt(gnn.f1)} vs. {fmt(xgb.f1)}).
				</p>
			{/if}
		{/if}
	{/if}
</div>
