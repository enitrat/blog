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
		/** work = employment (Zama, Kakarot); project = things built alongside it. */
		kind: z.enum(['work', 'project']).default('work'),
		role: z.string().optional(),
		/** Free-form, e.g. "2023 — present". */
		period: z.string().optional(),
		site: z.string().url().optional(),
		/** Sort key for the work indexes; lowest first. */
		order: z.number().default(99),
	}),
});

/**
 * A shelf-side reflection on one book. The filename IS the book's ISBN-13, so
 * a file existing here is the whole "this volume is annotated" signal — there
 * is no flag to keep in sync in `booksData.ts`. Substantial commentary on a
 * book belongs in `blog` and is reached through the book's `writingSlug`.
 */
const notes = defineCollection({
	loader: glob({ base: './src/content/notes', pattern: '**/*.{md,mdx}' }),
	schema: z.object({}),
});

export const collections = { blog, work, notes };
