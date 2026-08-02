<script lang="ts">
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import Controls from '$lib/components/Controls.svelte';
	import MetricsTable from '$lib/components/MetricsTable.svelte';
	import NodePanel from '$lib/components/NodePanel.svelte';
</script>

<svelte:head>
	<title>GraphSentry</title>
</svelte:head>

<div class="min-h-screen bg-slate-50">
	<header class="border-b border-slate-200 bg-white px-6 py-4">
		<h1 class="text-lg font-semibold text-slate-900">GraphSentry</h1>
		<p class="text-sm text-slate-500">
			Fraud detection on the Elliptic Bitcoin transaction network -- XGBoost baseline vs. GraphSAGE GNN.
		</p>
	</header>

	<main class="mx-auto max-w-7xl space-y-4 p-6">
		<Controls />

		<div class="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_20rem]">
			<div class="rounded-lg border border-slate-200 bg-white p-4">
				<h2 class="mb-3 text-sm font-semibold text-slate-900">Transaction network</h2>
				<NetworkGraph />
			</div>
			<NodePanel />
		</div>

		<MetricsTable />
	</main>
</div>
