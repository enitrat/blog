/**
 * Which `h2`s count as sections of the argument.
 *
 * Shared by the rehype plugin that numbers headings and hangs their margin
 * reference, and by the article layout that numbers the contents list. Both must
 * agree or the margin's §N and the index's §N stop naming the same place, so the
 * predicate lives here rather than being written twice.
 *
 * An article may contain its own apparatus as a heading — an inline "Table of
 * Contents", for instance. Numbering those makes the index list itself as a
 * section, which is the one case where the ordinal names nothing the reader
 * needs, so they are skipped.
 */
const APPARATUS_SLUGS = new Set([
	'table-of-contents',
	'contents',
	'toc',
	'sommaire',
	'table-des-matieres',
]);

export function isArgumentHeading(slug) {
	return !APPARATUS_SLUGS.has(String(slug ?? '').toLowerCase());
}
