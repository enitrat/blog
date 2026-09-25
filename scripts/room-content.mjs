/** The site content the room is baked from. `astro:content` is not available
 * out here, so the bake and its contract check read the collections directly. */
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const NOTES = 'src/content/notes';
const BLOG = 'src/content/blog';
const markdown = /\.mdx?$/;

/** ISBNs with a written note; a note's filename is its ISBN. */
export async function notedIsbns() {
	const files = await readdir(NOTES);
	return new Set(
		files.filter((file) => markdown.test(file)).map((file) => file.replace(markdown, '')),
	);
}

/** Published English writing, newest first. A piece is a directory in the blog
 * collection whose route is its directory name. */
export async function englishPieces() {
	const pieces = [];
	for (const directory of await readdir(BLOG, { withFileTypes: true })) {
		if (!directory.isDirectory()) continue;
		for (const file of (await readdir(join(BLOG, directory.name))).filter((name) =>
			markdown.test(name),
		)) {
			const source = await readFile(join(BLOG, directory.name, file), 'utf8');
			// Only three flat fields are read; a strict YAML parser rejects the
			// multi-line diagrams some posts carry in their frontmatter.
			const front = source.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
			const field = (name) =>
				front
					.match(new RegExp(`^${name}:\\s*(.+)$`, 'm'))?.[1]
					.trim()
					.replace(/^(['"])(.*)\1$/, '$2');
			if ((field('lang') ?? 'en') !== 'en') continue;
			const date = new Date(field('pubDate'));
			pieces.push({
				slug: directory.name,
				title: field('title'),
				// Dates are UTC midnight; format them in UTC so every machine bakes the same month.
				date: date.toLocaleDateString('en-GB', { month: 'long', year: 'numeric', timeZone: 'UTC' }),
				time: date.valueOf(),
			});
		}
	}
	return pieces.sort((a, b) => b.time - a.time);
}
