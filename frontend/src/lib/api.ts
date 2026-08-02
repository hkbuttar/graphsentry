/**
 * Typed fetch wrappers for the FastAPI backend (backend/routers/*.py).
 * Types here mirror backend/schemas.py exactly -- if a response shape
 * changes on the backend, it should change here too, not silently drift.
 *
 * All calls happen client-side (component onMount, not SvelteKit `load`
 * functions), since this is an inherently interactive, client-rendered
 * dashboard -- there's no benefit to SSR-fetching data that's about to be
 * driven by client-side sliders/toggles anyway, and it keeps the backend
 * URL out of the server-side rendering path entirely.
 */
import { PUBLIC_API_BASE_URL } from '$env/static/public';

export interface NetworkNode {
	id: string;
	time_step: number;
	label: number;
	xgboost_proba: number | null;
	gnn_proba: number | null;
	pagerank: number;
	community: number;
	in_degree: number;
	out_degree: number;
}

export interface NetworkEdge {
	source: string;
	target: string;
}

export interface NetworkResponse {
	time_step: number;
	nodes: NetworkNode[];
	edges: NetworkEdge[];
	truncated: boolean;
}

export interface ModelMetrics {
	model: string;
	split: string;
	n: number;
	precision: number;
	recall: number;
	f1: number;
	pr_auc: number;
}

export interface MetricsResponse {
	metrics: ModelMetrics[];
}

export interface NodeDetail {
	id: string;
	time_step: number;
	label: number;
	features: Record<string, number>;
	xgboost_proba: number | null;
	gnn_proba: number | null;
	in_neighbors: string[];
	out_neighbors: string[];
}

async function getJson<T>(path: string): Promise<T> {
	const response = await fetch(`${PUBLIC_API_BASE_URL}${path}`);
	if (!response.ok) {
		throw new Error(`${path} failed: ${response.status} ${response.statusText}`);
	}
	return response.json();
}

// 500, not the backend's own default of 300 -- the curated "interesting"
// time steps in Controls.svelte were deliberately chosen for high illicit
// counts (up to 342), so a 300 cap let illicit nodes alone fill the entire
// render budget, leaving zero room for a licit/unlabeled node to ever
// appear. 500 leaves headroom for all of them plus real color variety.
export function getNetwork(timeStep: number, maxNodes = 500): Promise<NetworkResponse> {
	return getJson(`/network?time_step=${timeStep}&max_nodes=${maxNodes}`);
}

export function getMetrics(): Promise<MetricsResponse> {
	return getJson('/metrics');
}

export function getNode(id: string): Promise<NodeDetail> {
	return getJson(`/node/${id}`);
}
