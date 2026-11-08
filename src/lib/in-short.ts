import type { ElementContent, Root, RootContent } from 'hast';

/** One entry of the "In short" block: an h2 and the first sentence of its first paragraph. */
export type InShortItem = { id: string; title: string; hook: string };

/** UI label of the block, also used by each section's link back to it. */
export const IN_SHORT_LABEL = 'In short';

/** Words whose trailing dot does not end a sentence. */
const ABBREV = new Set([
	'e.g',
	'i.e',
	'vs',
	'cf',
	'approx',
	'incl',
	'resp',
	'mr',
	'mrs',
	'dr',
	'st',
	'fig',
	'ex',
]);

/**
 * The "In short" items of a rendered post: each h2 with the verbatim first sentence of its first
 * paragraph, which may sit under an h3. `hooks` (frontmatter) overrides a hook by heading text.
 * A section with no paragraph gets an empty hook.
 */
export function inShort(tree: Root, hooks: Record<string, string> = {}): InShortItem[] {
	const items: InShortItem[] = [];
	let open = false;
	for (const node of tree.children) {
		if (node.type !== 'element') continue;
		if (node.tagName === 'h1' || node.tagName === 'h2') {
			open = node.tagName === 'h2';
			if (open) items.push({ id: String(node.properties.id), title: text(node), hook: '' });
			continue;
		}
		const item = items.at(-1);
		if (!open || !item || node.tagName !== 'p' || emphasisOnly(node)) continue;
		const paragraph = text(node).replace(/\s+/g, ' ').trim();
		if (!paragraph) continue;
		item.hook = firstSentence(paragraph);
		open = false;
	}
	return items.map((item) => ({ ...item, hook: hooks[item.title] ?? item.hook }));
}

/**
 * First sentence of `text`, verbatim. Never cuts at "e.g.", "vs.", "U.S.", a single initial
 * before a lowercase word, or "3.5".
 */
export function firstSentence(text: string): string {
	const end = /[.!?…]+["')\]”’]*(?=\s|$)/g;
	for (let m = end.exec(text); m; m = end.exec(text)) {
		const stop = m.index + m[0].length;
		if (m[0][0] === '.' && !m[0].startsWith('..')) {
			const word = (text.slice(0, m.index).split(/\s/).pop() ?? '')
				.toLowerCase()
				.replace(/^[("'“‘]+/, '');
			const next = text.slice(stop).trimStart();
			const capital = /^[A-Z]/.test(next);
			if (word === 'etc') {
				if (!capital) continue;
			} else if (ABBREV.has(word) || /^(?:[a-z]\.)+[a-z]$/.test(word)) {
				continue;
			} else if (/^[a-z]$/.test(word) && (next.startsWith('.') || /^[a-z]/.test(next))) {
				continue;
			}
		}
		return text.slice(0, stop);
	}
	return text;
}

/** Text of a node, without footnote reference numbers. */
function text(node: RootContent | ElementContent): string {
	if (node.type === 'text') return node.value;
	if (node.type !== 'element') return '';
	if (node.tagName === 'sup' && node.children.some(isFootnoteRef)) return '';
	return node.children.map(text).join('');
}

function isFootnoteRef(node: ElementContent): boolean {
	return node.type === 'element' && node.properties.dataFootnoteRef !== undefined;
}

/** A paragraph that is one emphasis span, such as "_This was first published elsewhere._". */
function emphasisOnly(node: { children: ElementContent[] }): boolean {
	const parts = node.children.filter((c) => c.type !== 'text' || c.value.trim());
	return (
		parts.length === 1 &&
		parts[0].type === 'element' &&
		(parts[0].tagName === 'em' || parts[0].tagName === 'strong')
	);
}
