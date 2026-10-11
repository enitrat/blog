import type { MarkdownHeading } from 'astro';

export type InShortItem = { id: string; title: string; hook: string };

/** UI label of the block, also used by each section's link back to it. */
export const IN_SHORT_LABEL = 'In short';

// Words whose trailing dot does not end a sentence.
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

/** Plain text from one markdown paragraph. */
export function stripMarkdown(md: string): string {
	return md
		.replace(/!\[[^\]]*\]\([^)]*\)/g, '')
		.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
		.replace(/<[^>]+>/g, '')
		.replace(/`([^`]*)`/g, '$1')
		.replace(/(\*\*|__)(.+?)\1/g, '$2')
		.replace(/\*([^*\s][^*]*?)\*/g, '$1')
		.replace(/(^|\W)_([^_]+?)_(?=\W|$)/g, '$1$2')
		.replace(/\s+/g, ' ')
		.trim();
}

/**
 * First sentence of `text`, verbatim. Never cuts at "e.g.", "vs.", "U.S.", a
 * single initial before a lowercase word, or "3.5".
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

const INLINE_HTML =
	/^<\/?(kbd|mark|em|strong|b|i|code|span|a|sub|sup|abbr|small|u|s|del|ins)[\s>/]/i;
const NOT_PARAGRAPH =
	/^(\s*([-*+]|\d+[.)])\s|\s*>|\s*\||\s*!\[|\s*(-{3,}|\*{3,}|_{3,})\s*$|\[\^[^\]]+\]:| {4,}|\t)/;
const EMPHASIS_ONLY = /^([*_]{1,2})[^*_]+\1$/;

/** The first real paragraph (plain text) of each h2 in a body, in document order; '' when none. */
function sectionParagraphs(body: string): string[] {
	const out: string[] = [];
	let fence = false;
	let open = false; // current h2 still wants a paragraph
	let skip = false; // inside a list, quote, table, footnote or HTML block, until a blank line
	let para: string[] = [];
	const flush = () => {
		const raw = para.join(' ').trim();
		para = [];
		if (!open || !raw || EMPHASIS_ONLY.test(raw)) return;
		out[out.length - 1] = stripMarkdown(raw);
		open = false;
	};
	for (const line of body.split(/\r?\n/)) {
		if (/^\s*(```|~~~)/.test(line)) {
			flush();
			fence = !fence;
			continue;
		}
		if (fence) continue;
		if (!line.trim()) {
			flush();
			skip = false;
			continue;
		}
		if (skip) continue;
		const h = /^(#{1,6})\s+(.+?)\s*#*$/.exec(line);
		if (h) {
			flush();
			// h3 and deeper keep the h2's slot open: its first paragraph may sit under a subheading.
			if (h[1].length <= 2) open = h[1] === '##';
			if (h[1] === '##') out.push('');
			continue;
		}
		const html = line.trimStart().startsWith('<') && !INLINE_HTML.test(line.trimStart());
		if (
			html ||
			(!para.length && NOT_PARAGRAPH.test(line)) ||
			/^\s*(-{3,}|\[\^[^\]]+\]:)/.test(line)
		) {
			flush();
			skip = true;
		} else if (para.length && /^\s*([-*+]|\d+[.)])\s/.test(line)) {
			flush();
			skip = true;
		} else para.push(line.trim());
	}
	flush();
	return out;
}

/**
 * The "In short" items of a blog entry: each h2 with the verbatim first sentence
 * of its first paragraph, matched by order. `hooks` (frontmatter) overrides a
 * hook by the rendered heading text.
 */
export function inShort(
	entry: { body?: string; data: { hooks?: Record<string, string> } },
	headings: MarkdownHeading[],
): InShortItem[] {
	const paragraphs = sectionParagraphs(entry.body ?? '');
	return headings
		.filter((h) => h.depth === 2)
		.map(({ slug, text }, i) => ({
			id: slug,
			title: text,
			hook: entry.data.hooks?.[text] ?? firstSentence(paragraphs[i] ?? ''),
		}));
}
