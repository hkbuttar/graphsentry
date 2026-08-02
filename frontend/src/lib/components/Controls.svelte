<script lang="ts">
	/**
	 * All three interactive controls in one place -- each just reads/writes a
	 * shared store. NetworkGraph reacts to all three independently; nothing
	 * here needs to know that NetworkGraph exists.
	 */
	import { activeModel, probabilityThreshold, activeTimeStep } from '$lib/stores';

	// Time steps with a meaningfully large illicit count, so the picker
	// doesn't dump the user into a mostly-empty view -- see backend's
	// /network default (32) and the README's per-time-step illicit table.
	const interestingTimeSteps = [32, 29, 13, 20, 9, 42, 35, 22, 15, 24];
</script>

<div class="flex flex-wrap items-center gap-6 rounded-lg border border-slate-200 bg-white p-4 text-sm">
	<label class="flex items-center gap-2">
		<span class="text-slate-600">Model</span>
		<select
			class="rounded border border-slate-300 px-2 py-1"
			bind:value={$activeModel}
		>
			<option value="xgboost">XGBoost</option>
			<option value="gnn">GraphSAGE</option>
		</select>
	</label>

	<label class="flex items-center gap-2">
		<span class="text-slate-600">Time step</span>
		<select
			class="rounded border border-slate-300 px-2 py-1"
			bind:value={$activeTimeStep}
		>
			{#each interestingTimeSteps as step (step)}
				<option value={step}>{step}</option>
			{/each}
		</select>
	</label>

	<label class="flex flex-1 items-center gap-2 min-w-[16rem]">
		<span class="text-slate-600 whitespace-nowrap">
			Illicit threshold: {$probabilityThreshold.toFixed(2)}
		</span>
		<input
			type="range"
			min="0"
			max="1"
			step="0.01"
			bind:value={$probabilityThreshold}
			class="flex-1 accent-red-600"
		/>
	</label>
</div>
