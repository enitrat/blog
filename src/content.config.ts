import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
	// Load Markdown and MDX files in the `src/content/blog/` directory.
	loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
	// Type-check frontmatter using a schema
	schema: z.object({
		title: z.string(),
		description: z.string(),
		// Transform string to Date object
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		lang: z.enum(['en', 'fr']).default('en'),
		/** Base slugs of sibling pieces to offer in the close instead of the chronological neighbours. */
		related: z.array(z.string()).default([]),
	}),
});

const work = defineCollection({
	loader: glob({ base: './src/content/work', pattern: '**/*.{md,mdx}' }),
	schema: z.object({
		title: z.string(),
		summary: z.string(),
		role: z.string().optional(),
		/** Free-form, e.g. "2023 — present". */
		period: z.string().optional(),
		site: z.string().url().optional(),
		/** Sort key for the work indexes; lowest first. */
		order: z.number().default(99),
	}),
});

export const collections = { blog, work };
