import 'server-only';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import matter from 'gray-matter';
import type { StaticImageData } from 'next/image';
import { cache } from 'react';
import { z } from 'zod';
import { findImage } from './media';
import { smartQuotes } from './typeset';

/** Frontmatter prose: typographer's quotes, as the markdown bodies get. */
const text = z.string().transform(smartQuotes);

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
	title: text,
	description: text,
	pubDate: z.coerce.date(),
	updatedDate: z.coerce.date().optional(),
	/** Base slugs of sibling pieces to offer in the close instead of the chronological neighbours. */
	related: z.array(z.string()).default([]),
	/** Overrides for the "In short" list: heading text -> sentence. Default is each section's first sentence. */
	hooks: z.record(z.string(), text).optional(),
});

const workSchema = z.object({
	title: text,
	summary: text,
	/** work = employment (Zama, Kakarot); project = things built alongside it. */
	kind: z.enum(['work', 'project']).default('work'),
	role: text.optional(),
	/** Free-form, such as "2023 — present". */
	period: z.string().optional(),
	site: z.url().optional(),
	/** Sort key for the work indexes; lowest first. */
	order: z.number().default(99),
});

const tripSchema = z.object({
	title: text,
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

/** Essays, newest first. */
export const getPosts = cache((): Post[] =>
	folders('blog', blogSchema).sort((a, b) => b.data.pubDate.getTime() - a.data.pubDate.getTime()),
);

/** Work and projects, by their `order`. */
export const getReports = cache((): Report[] =>
	files('work', workSchema).sort((a, b) => a.data.order - b.data.order),
);

/** Reading notes, one per book, named by its ISBN-13. */
export const getNotes = cache(() => files('notes', z.object({}).strict()));

/** Trips, newest first. Empty until the first `src/content/trips/<slug>/index.md` exists. */
export const getTrips = cache((): Trip[] =>
	folders('trips', tripSchema).sort((a, b) => b.data.date.getTime() - a.data.date.getTime()),
);

/** The one place that knows the route shape of an essay. */
export const postUrl = (post: Pick<Post, 'id'>) => `/writing/${post.id}/`;

export const workUrl = (report: Pick<Report, 'id'>) => `/work/${report.id}/`;

type PostLink = { href: string; title: string };
const linkTo = (post: Post): PostLink => ({ href: postUrl(post), title: post.data.title });

/** The post's `related` slugs as links, dropping any that name nothing. */
export function getRelated(post: Post): PostLink[] {
	const posts = getPosts();
	return post.data.related.flatMap((slug) => posts.filter((p) => p.id === slug).map(linkTo));
}

/** Chronological neighbours, so an essay offers a way onward rather than only a way back. */
export function getAdjacent(post: Post): { newer?: PostLink; older?: PostLink } {
	const posts = getPosts();
	const index = posts.findIndex((p) => p.id === post.id);
	const newer = posts[index - 1];
	const older = posts[index + 1];
	return { newer: newer && linkTo(newer), older: older && linkTo(older) };
}

/** The post's `banner.*` image from its folder, if it has one. */
export const bannerFor = (post: Pick<Post, 'id'>): Promise<StaticImageData | undefined> =>
	findImage(`content/blog/${post.id}/banner`);
