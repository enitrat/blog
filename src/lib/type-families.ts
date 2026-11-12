/**
 * The site's three typefaces, named once. `layout.tsx` loads each with next/font (which takes
 * literal options only), `globals.css` maps its variable onto a family token, the OG cards draw it
 * from `src/assets/fonts/`, and `design-tokens.test.ts` checks that all of them agree with this list.
 */
export const TYPE_FAMILIES = [
	{
		token: '--font-sans',
		variable: '--font-instrument-sans',
		name: 'Instrument Sans',
		job: 'Headings and interface',
	},
	{
		token: '--font-serif',
		variable: '--font-source-serif',
		name: 'Source Serif 4',
		job: 'Reading',
	},
	{
		token: '--font-mono',
		variable: '--font-fragment-mono',
		name: 'Fragment Mono',
		job: 'Meta, figures and code',
	},
] as const;

export type TypeFamily = (typeof TYPE_FAMILIES)[number];

export const familyOf = (token: TypeFamily['token']) =>
	TYPE_FAMILIES.find((f) => f.token === token) as TypeFamily;
