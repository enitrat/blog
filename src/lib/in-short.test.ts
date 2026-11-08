import { expect, test } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import matter from 'gray-matter';
import { firstSentence, inShort } from './in-short';
import { renderMarkdown } from './markdown';

const items = async (body: string, hooks?: Record<string, string>) =>
	inShort(await renderMarkdown(body, '.'), hooks);
const hooks = async (body: string) => (await items(body)).map((i) => i.hook);

test('does not cut at abbreviations or decimals', () => {
	expect(firstSentence('Use tools, e.g. a linter, vs. nothing. Next one.')).toBe(
		'Use tools, e.g. a linter, vs. nothing.',
	);
	expect(firstSentence('It costs 3.5 times more, i.e. a lot! Yes.')).toBe(
		'It costs 3.5 times more, i.e. a lot!',
	);
});

test('a year before a capital ends the sentence', () => {
	expect(firstSentence('It shipped in 2024. Then we left.')).toBe('It shipped in 2024.');
});

test('single letters and dotted initials', () => {
	expect(firstSentence('Take plan A. Then plan B.')).toBe('Take plan A.');
	expect(firstSentence('Vitamin C. and more. Next.')).toBe('Vitamin C. and more.');
	expect(firstSentence('The U.S. Then we went. Next.')).toBe('The U.S. Then we went.');
});

test('"no." ends a sentence; "etc." only before a capital', () => {
	expect(firstSentence('I said no. Then left.')).toBe('I said no.');
	expect(firstSentence('Tools, etc. are fine. Next.')).toBe('Tools, etc. are fine.');
	expect(firstSentence('We use tea, coffee, etc. Then we left.')).toBe('We use tea, coffee, etc.');
});

test('plain text from markdown, skips images and lists, honours hooks', async () => {
	const body =
		'## One\n\n![](https://example.com/a.png)\n\n- item\n\nHello _there_ `x`. More.\n\n## Two\n\nSecond one. Rest.\n';
	expect(await items(body, { Two: 'Mine.' })).toEqual([
		{ id: 'one', title: 'One', hook: 'Hello there x.' },
		{ id: 'two', title: 'Two', hook: 'Mine.' },
	]);
});

test('hooks are keyed by the rendered heading text', async () => {
	const body =
		"## Don't stop\n\nFirst. x\n\n## A &amp; B\n\nSecond. y\n\n## <mark>Hot</mark>\n\nThird. z\n";
	expect((await items(body, { 'Don’t stop': 'Override.' })).map((i) => i.hook)).toEqual([
		'Override.',
		'Second.',
		'Third.',
	]);
});

test('duplicate titles each get their own hook', async () => {
	expect(await hooks('## Same\n\nOne. a\n\n## Same\n\nTwo. b\n')).toEqual(['One.', 'Two.']);
});

test('wrapped list items belong to the list', async () => {
	expect(await hooks('## A\n\n- item\n  continues here. Not me.\n\nReal. Rest.\n')).toEqual([
		'Real.',
	]);
});

test('inline HTML still opens a paragraph; block HTML does not', async () => {
	expect(await hooks('## A\n\n<kbd>Esc</kbd> closes it. Rest.\n')).toEqual(['Esc closes it.']);
	expect(await hooks('## A\n\n<div>\nx\n</div>\n\nReal. Rest.\n')).toEqual(['Real.']);
});

test('footnotes are never hooks, nor part of one', async () => {
	expect(await hooks('## A\n\nReal[^1]. Rest.\n\n[^1]: A note.\n')).toEqual(['Real.']);
});

test('a paragraph under a subheading counts for its h2', async () => {
	expect(await hooks('## A\n\n### Sub\n\nUnder sub. Rest.\n\n## B\n\n## C\n\nCee.\n')).toEqual([
		'Under sub.',
		'',
		'Cee.',
	]);
});

test('a leading all-emphasis paragraph is skipped', async () => {
	expect(await hooks('## A\n\n_This was written elsewhere._\n\nReal one. Rest.\n')).toEqual([
		'Real one.',
	]);
	expect(await hooks('## A\n\n*Only emphasis.*\n')).toEqual(['']);
});

/** Real posts: the sections that truly have no paragraph before their next heading. */
const NO_PARAGRAPH: Record<string, string[]> = {
	'cairo-1-sierra-1': ['Part 1: An essential layer to produce safe CASM'],
	'cairo-1-sierra-2': ['Part 2: Anatomy of Sierra programs'],
	'cairo-1-sierra-3': ['Part 3: Become a better Cairo developer with Sierra'],
};
const blog = new URL('../content/blog/', import.meta.url);
const posts = readdirSync(blog, { withFileTypes: true }).filter((d) => d.isDirectory());

test.each(posts.map((d) => d.name))('real content: %s', async (name) => {
	const { content } = matter(readFileSync(new URL(`${name}/index.md`, blog), 'utf8'));
	const tree = await renderMarkdown(content, `blog/${name}`);
	const empty = inShort(tree)
		.filter((i) => !i.hook)
		.map((i) => i.title);
	expect(empty).toEqual(NO_PARAGRAPH[name] ?? []);
});
