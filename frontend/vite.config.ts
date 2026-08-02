import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-vercel';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},

			// Explicit adapter-vercel rather than adapter-auto: this project deploys
			// to Vercel specifically (see README Deployment), and an explicit adapter
			// avoids relying on adapter-auto's dynamic platform detection resolving
			// the right package correctly on a first deploy.
			adapter: adapter()
		})
	]
});
