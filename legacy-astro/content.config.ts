import { defineCollection, z } from 'astro:content';
import { existsSync, readdirSync } from 'node:fs';
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
		/** Base slugs of sibling pieces to offer in the close instead of the chronological neighbours. */
		related: z.array(z.string()).default([]),
		/** Overrides for the "In short" list: heading text -> sentence. Default is each section's first sentence. */
		hooks: z.record(z.string()).optional(),
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

/**
 * One trip: a folder `src/content/trips/<slug>/` holding `index.md` and its photos
 * (see the README there). Sorted by `date`, newest first, on /places.
 */
const trips = defineCollection({
	loader: glob({ base: './src/content/trips', pattern: '*/index.md' }),
	schema: ({ image }) =>
		z.object({
			/** The place. */
			title: z.string(),
			country: z.string(),
			date: z.coerce.date(),
			/** [lat, lon]. */
			coords: z.tuple([z.number(), z.number()]).optional(),
			cover: image(),
			photos: z.array(z.object({ src: image(), caption: z.string().optional() })),
		}),
});

// With no trip folder the glob loader warns on every build, so define the collection only once
// one exists. `getTrips()` (src/utils/trips.ts) is the one reader and returns [] meanwhile.
const hasTrips = readdirSync('./src/content/trips', { withFileTypes: true }).some(
	(d) => d.isDirectory() && existsSync(`./src/content/trips/${d.name}/index.md`),
);

export const collections = { blog, work, notes, ...(hasTrips ? { trips } : {}) };
