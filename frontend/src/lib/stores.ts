/**
 * Shared, cross-component reactive state, built with Svelte's `writable`/
 * `derived` stores rather than component-local state.
 *
 * This is a deliberate architectural choice worth naming explicitly: unlike
 * React, where shared state either gets lifted into a common ancestor and
 * threaded down through props, or reached for via a separate library
 * (Context, Redux, Zustand), Svelte stores are values that live outside the
 * component tree entirely. Any component can `import` and subscribe to one
 * directly -- no prop-drilling, no provider wrapper, no separate state
 * library. Subscribing is as simple as prefixing the store name with `$` in
 * a component's markup or script (e.g. `$selectedNodeId`), and Svelte
 * compiles that into the subscribe/unsubscribe boilerplate automatically.
 *
 * Four independent pieces of state live here because multiple, unrelated
 * components all need to react to them: the network graph, the node lookup
 * panel, and the probability slider all care about `selectedNodeId`; the
 * network graph and the model toggle both care about `activeModel` and
 * `probabilityThreshold`.
 */
import { writable } from 'svelte/store';

export const selectedNodeId = writable<string | null>(null);

export const activeModel = writable<'xgboost' | 'gnn'>('xgboost');

/** Nodes with predicted probability >= this, under the active model, are
 * highlighted as "predicted illicit" in the network view. */
export const probabilityThreshold = writable(0.5);

/** Which time step's subgraph the network view is currently showing --
 * defaults to the backend's own default (see backend/routers/network.py). */
export const activeTimeStep = writable(32);
