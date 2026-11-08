import 'server-only';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import matter from 'gray-matter';
import { z } from 'zod';

/** The markdown collections, read once per build from `src/content/`. Replaces Astro's content layer. */
const CONTENT = join(process.cwd(), 'src/content');

export type Entry<T> = {
	/** The folder name (`blog/<id>/index.md`) or the file name (`work/<id>.md`). */
	id: string;
	data: T;
	/** The markdown body, frontmatter removed. */
	body: string;
	/** Directory of the entry, relative to `src/content`, for resolving relative images. */
	dir: string;
};

const blogSchema = z.object({
	title: z.string(),
	description: z.string(),
	pubDate: z.coerce.date(),
	updatedDate: z.coerce.date().optional(),
	/** Base slugs of sibling pieces to offer in the close instead of the chronological neighbours. */
	related: z.array(z.string()).default([]),
	/** Overrides for the "In short" list: heading text -> sentence. Default is each section's first sentence. */
	hooks: z.record(z.string(), z.string()).optional(),
});

const workSchema = z.object({
	title: z.string(),
	summary: z.string(),
	/** work = employment (Zama, Kakarot); project = things built alongside it. */
	kind: z.enum(['work', 'project']).default('work'),
	role: z.string().optional(),
	/** Free-form, such as "2023 — present". */
	period: z.string().optional(),
	site: z.url().optional(),
	/** Sort key for the work indexes; lowest first. */
	order: z.number().default(99),
});

const tripSchema = z.object({
	title: z.string(),
	country: z.string(),
	date: z.coerce.date(),
	/** [lat, lon]. */
	coords: z.tuple([z.number(), z.number()]).optional(),
	/** Relative to the trip folder, such as `./cover.jpg`. */
	cover: z.string(),
	photos: z.array(z.object({ src: z.string(), caption: z.string().optional() })),
});

export type BlogData = z.infer<typeof blogSchema>;
export type WorkData = z.infer<typeof workSchema>;
export type TripData = z.infer<typeof tripSchema>;
export type Post = Entry<BlogData>;
export type Report = Entry<WorkData>;
export type Trip = Entry<TripData>;

function parse<T>(schema: z.ZodType<T>, file: string, id: string, dir: string): Entry<T> {
	const { data, content } = matter(readFileSync(file, 'utf8'));
	const result = schema.safeParse(data);
	if (!result.success) throw new Error(`${file}: ${z.prettifyError(result.error)}`);
	return { id, data: result.data, body: content, dir };
}

/** `<collection>/<id>/index.md` entries. */
function folders<T>(collection: string, schema: z.ZodType<T>): Entry<T>[] {
	const root = join(CONTENT, collection);
	if (!existsSync(root)) return [];
	return readdirSync(root, { withFileTypes: true })
		.filter((d) => d.isDirectory() && existsSync(join(root, d.name, 'index.md')))
		.map((d) => parse(schema, join(root, d.name, 'index.md'), d.name, `${collection}/${d.name}`));
}

/** `<collection>/<id>.md` entries. */
function files<T>(collection: string, schema: z.ZodType<T>): Entry<T>[] {
	const root = join(CONTENT, collection);
	return readdirSync(root, { withFileTypes: true })
		.filter((d) => d.isFile() && /\.mdx?$/.test(d.name) && d.name !== 'README.md')
		.map((d) => parse(schema, join(root, d.name), d.name.replace(/\.mdx?$/, ''), collection));
}

const cache = new Map<string, unknown>();
const once = <T>(key: string, load: () => T): T => {
	if (!cache.has(key)) cache.set(key, load());
	return cache.get(key) as T;
};

export const getPosts = (): Post[] => once('blog', () => folders('blog', blogSchema));
export const getReports = (): Report[] => once('work', () => files('work', workSchema));
export const getNotes = (): Entry<Record<string, never>>[] =>
	once('notes', () => files('notes', z.object({}).strict() as never));
/** Trips, newest first. Empty until the first `src/content/trips/<slug>/index.md` exists. */
export const getTrips = (): Trip[] =>
	once('trips', () =>
		folders('trips', tripSchema).sort((a, b) => b.data.date.getTime() - a.data.date.getTime()),
	);
