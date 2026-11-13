/**
 * Typesetting for text that bypasses the markdown pipeline: frontmatter titles, descriptions and
 * roles. Essay bodies already get remark-smartypants. Only spacing and quote characters change,
 * never the words.
 */
export function typeset(text: string) {
	return bindNumbers(smartQuotes(text));
}

/** Straight quotes to typographer's quotes. */
export function smartQuotes(text: string) {
	return text
		.replace(/(^|[\s([{—-])"/g, '$1“')
		.replace(/"/g, '”')
		.replace(/(^|[\s([{“—-])'/g, '$1‘')
		.replace(/'/g, '’');
}

/**
 * A number never starts a line: the space before it becomes a no-break space, so "Cairo 1.0" and
 * "Part 3" wrap as one unit wherever the text is set, with or without `text-wrap: balance`.
 */
export function bindNumbers(text: string) {
	return text.replace(/ (?=\d)/g, ' ');
}
