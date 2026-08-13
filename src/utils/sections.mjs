/*
 * Section numbering for articles: which `h2`s count as sections of the
 * argument, and the rehype plugin that stamps each one with its §ordinal.
 *
 * The article layout numbers its contents list with the same predicate. Both
 * must agree or the margin's §N and the index's §N stop naming the same place,
 * which is why the predicate and the plugin live together here.
 */

/**
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

/** Numbers every argument `h2` and hangs a §ordinal link in the outer margin. */
export function rehypeSectionMarks() {
	return (tree) => {
		let n = 0;
		const walk = (node) => {
			if (!node.children) return;
			for (const child of node.children) {
				if (
					child.type === 'element' &&
					child.tagName === 'h2' &&
					child.properties?.id &&
					isArgumentHeading(child.properties.id)
				) {
					n += 1;
					child.properties['data-section'] = String(n);
					child.children.unshift({
						type: 'element',
						tagName: 'a',
						properties: {
							className: ['section-mark'],
							href: `#${child.properties.id}`,
							// The glyph and numeral below are decorative duplicates of the
							// adjacent heading, so the link needs its own name.
							'aria-label': `Link to section ${n}`,
						},
						children: [
							{
								type: 'element',
								tagName: 'span',
								properties: {},
								children: [{ type: 'text', value: '§' }],
							},
							{
								type: 'element',
								tagName: 'b',
								properties: {},
								children: [{ type: 'text', value: String(n) }],
							},
						],
					});
				}
				walk(child);
			}
		};
		walk(tree);
	};
}
