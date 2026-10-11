import { expect, test } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { firstSentence, inShort, stripMarkdown } from './inShort';

const h2 = (...texts: string[]) => texts.map((text, i) => ({ depth: 2, slug: `s${i}`, text }));
const hooks = (body: string, ...titles: string[]) =>
	inShort({ body, data: {} }, h2(...titles)).map((i) => i.hook);

test('does not cut at abbreviations or decimals', () => {
	expect(firstSentence('Use tools, e.g. a linter, vs. nothing. Next one.')).toBe(
		'Use tools, e.g. a linter, vs. nothing.',
	);
	expect(firstSentence('It costs 3.5 times more, i.e. a lot! Yes.')).toBe(
		'It costs 3.5 times more, i.e. a lot!',
	);
});

test('strips markdown', () => {
	expect(stripMarkdown('a _b_ **c** [d](http://x) `e` snake_case')).toBe('a b c d e snake_case');
});

test('extracts, skips images and lists, honours hooks', () => {
	const body =
		'## One\n\n![](./a.png)\n\n- item\n\nHello _there_. More.\n\n## Two\n\nSecond one. Rest.\n';
	const headings = h2('One', 'Two').map((h, i) => ({ ...h, slug: ['one', 'two'][i] }));
	expect(inShort({ body, data: { hooks: { Two: 'Mine.' } } }, headings)).toEqual([
		{ id: 'one', title: 'One', hook: 'Hello there.' },
		{ id: 'two', title: 'Two', hook: 'Mine.' },
	]);
});

test('matches by order: smartypants, entities and marks in headings do not miss', () => {
	const body =
		"## Don't stop\n\nFirst. x\n\n## A &amp; B\n\nSecond. y\n\n## <mark>Hot</mark>\n\nThird. z\n";
	const items = inShort(
		{ body, data: { hooks: { 'Don’t stop': 'Override.' } } },
		h2('Don’t stop', 'A & B', 'Hot'),
	);
	expect(items.map((i) => i.hook)).toEqual(['Override.', 'Second.', 'Third.']);
});

test('duplicate titles each get their own hook', () => {
	expect(hooks('## Same\n\nOne. a\n\n## Same\n\nTwo. b\n', 'Same', 'Same')).toEqual([
		'One.',
		'Two.',
	]);
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

test('wrapped list items belong to the list', () => {
	expect(hooks('## A\n\n- item\n  continues here. Not me.\n\nReal. Rest.\n', 'A')).toEqual([
		'Real.',
	]);
	expect(hooks('## A\n\n- item\ncontinues. Not me.\n\nReal. Rest.\n', 'A')).toEqual(['Real.']);
});

test('inline HTML still opens a paragraph; block HTML does not', () => {
	expect(hooks('## A\n\n<kbd>Esc</kbd> closes it. Rest.\n', 'A')).toEqual(['Esc closes it.']);
	expect(hooks('## A\n\n<mark>Key</mark> idea. Rest.\n', 'A')).toEqual(['Key idea.']);
	expect(hooks('## A\n\n<div>\nx\n</div>\n\nReal. Rest.\n', 'A')).toEqual(['Real.']);
});

test('footnote definitions are never hooks', () => {
	expect(hooks('## A\n\n[^1]: A note.\n\nReal. Rest.\n', 'A')).toEqual(['Real.']);
});

test('a paragraph under a subheading counts for its h2', () => {
	expect(
		hooks('## A\n\n### Sub\n\nUnder sub. Rest.\n\n## B\n\n## C\n\nCee.\n', 'A', 'B', 'C'),
	).toEqual(['Under sub.', '', 'Cee.']);
});

test('a leading all-emphasis paragraph is skipped', () => {
	expect(hooks('## A\n\n_This was written elsewhere._\n\nReal one. Rest.\n', 'A')).toEqual([
		'Real one.',
	]);
	expect(hooks('## A\n\n*Only emphasis.*\n', 'A')).toEqual(['']);
});

// Real posts. Sections that truly have no paragraph before their next heading.
const NO_PARAGRAPH: Record<string, string[]> = {
	'cairo-1-sierra-1': ['Part 1: An essential layer to produce safe CASM'],
	'cairo-1-sierra-2': ['Part 2: Anatomy of Sierra programs'],
	'cairo-1-sierra-3': ['Part 3: Become a better Cairo developer with Sierra'],
};
const dir = new URL('../content/blog/', import.meta.url);
const posts = readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory());

test.each(posts.map((d) => d.name))('real content: %s', (name) => {
	const src = readFileSync(new URL(`${name}/index.md`, dir), 'utf8');
	const body = src.replace(/^---\n[\s\S]*?\n---\n/, '');
	const titles = [
		...body.replace(/^(```|~~~)[\s\S]*?^\1.*$/gm, '').matchAll(/^## (.+?)\s*$/gm),
	].map((m) => stripMarkdown(m[1]));
	const items = inShort({ body, data: {} }, h2(...titles));
	console.log(`\n${name}`);
	for (const i of items) console.log(`  ${i.title} -> ${i.hook.slice(0, 80)}`);
	const empty = items.filter((i) => !i.hook).map((i) => i.title);
	expect(empty).toEqual(NO_PARAGRAPH[name] ?? []);
});
