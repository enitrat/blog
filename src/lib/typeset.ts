/**
 * Straight quotes to typographer's quotes, for text that bypasses the markdown pipeline:
 * frontmatter titles, descriptions and roles. Essay bodies already get remark-smartypants.
 * Only the characters change, never the words.
 */
export function smartQuotes(text: string) {
	return text
		.replace(/(^|[\s([{—-])"/g, '$1“')
		.replace(/"/g, '”')
		.replace(/(^|[\s([{“—-])'/g, '$1‘')
		.replace(/'/g, '’');
}
