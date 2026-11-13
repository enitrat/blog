import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { Glob } from 'bun';
import { offenders } from './source-lines';
import { TYPE_FAMILIES } from './type-families';

/** The one page that shows the palettes themselves. */
const SPECIMEN = 'src/app/design/page.tsx';

const COLOR_UTILITY =
	'(?:bg|text|border|ring|ring-offset|outline|fill|stroke|from|via|to|decoration|divide|shadow|placeholder|caret|accent)';

/** A class token, with any variants in front, that does not belong to the typography plugin. */
const utility = (body: string) =>
	new RegExp(`(?<![\\w:/-])(?!prose-)(?:[\\w\\[\\]&>.=/*()@-]+:)*${body}(?![\\w-])`);

describe('components name roles, not palettes or themes', () => {
	test('no palette outside globals.css and the design page', () => {
		expect(offenders(/--palette-/, (file) => file === SPECIMEN)).toEqual([]);
	});

	test('no stock or hue-named colour scale in a class', () => {
		const scale = new RegExp(
			`\\b${COLOR_UTILITY}-(?:red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|slate|gray|zinc|neutral|stone|paper|oxblood|brass|bottle)-\\d`,
		);
		expect(offenders(scale)).toEqual([]);
	});

	test('no dark: colour variant, since roles swap with the theme', () => {
		const darkColor = new RegExp(`dark:(?:[\\w-]+:)*${COLOR_UTILITY}-(?!\\(--shiki)`);
		expect(offenders(darkColor)).toEqual([]);
	});
});

describe('text names a type role', () => {
	test('no font size utility', () => {
		expect(offenders(utility('text-(?:xs|sm|base|lg|xl|[2-9]xl|\\[[^\\]]+\\])(?:/\\S+)?'))).toEqual(
			[],
		);
	});

	test('no font family, weight, tracking or leading utility', () => {
		const raw = utility(
			'(?:font-(?:sans|serif|mono|thin|extralight|light|normal|medium|semibold|bold|extrabold|black)|tracking-\\S+|leading-\\S+)',
		);
		expect(offenders(raw)).toEqual([]);
	});
});

describe('type decisions live in the roles', () => {
	test('no arbitrary family, weight or size, and no figure, case or style utility', () => {
		const raw = utility(
			'(?:font-[\\[(]\\S*|text-[\\[(](?:length|size):\\S*|tabular-nums|oldstyle-nums|lining-nums|proportional-nums|slashed-zero|diagonal-fractions|italic|not-italic|uppercase|lowercase|capitalize|normal-case)',
		);
		// Comments, next/font's `style: ['normal', 'italic']` and string comparisons are not classes.
		const notClasses = (hit: string) => /: (?:\*|\/\/)|style: \[|=== '/.test(hit);
		expect(offenders(raw).filter((hit) => !notClasses(hit))).toEqual([]);
	});

	test('no arbitrary type property', () => {
		expect(
			offenders(
				/\[(?:font-[a-z-]+|letter-spacing|line-height|text-transform|font-variant-numeric):/,
			),
		).toEqual([]);
	});

	test('no type in inline styles, except the OG card, which Satori draws without CSS', () => {
		expect(
			offenders(
				/\b(?:fontSize|fontWeight|fontFamily|fontStyle|fontVariant\w*|lineHeight|letterSpacing|textTransform)\s*:/,
				(file) => file === 'src/lib/og.tsx',
			),
		).toEqual([]);
	});

	test('no type in prose modifiers; the essay utility owns essay type', () => {
		expect(
			offenders(
				/prose-[\w-]+:(?:[\w[\]&>.=/*()@-]+:)*(?:type-|font-|italic|not-italic|leading-|tracking-|uppercase|normal-case|text-(?:xs|sm|base|lg|\d?xl)\b)/,
			),
		).toEqual([]);
	});

	test('globals.css is the only stylesheet', () => {
		expect([...new Glob('src/**/*.css').scanSync('.')]).toEqual(['src/app/globals.css']);
	});
});

describe('the type families agree with src/lib/type-families.ts', () => {
	const css = readFileSync('src/app/globals.css', 'utf8');
	const layout = readFileSync('src/app/layout.tsx', 'utf8');

	test('layout.tsx loads each family and globals.css maps it onto its token', () => {
		for (const family of TYPE_FAMILIES) {
			expect(layout).toContain(`variable: '${family.variable}'`);
			expect(css).toContain(`${family.token}: var(${family.variable})`);
		}
	});

	test('every OG font file is a cut of a site family', () => {
		const stems = TYPE_FAMILIES.map((f) => f.name.replaceAll(' ', ''));
		const stale = [...new Glob('src/assets/fonts/*').scanSync('.')].filter(
			(file) => !stems.some((stem) => file.startsWith(`src/assets/fonts/${stem}-`)),
		);
		expect(stale).toEqual([]);
	});

	test('the design page specimens every type role', () => {
		const page = readFileSync(SPECIMEN, 'utf8');
		const roles = [...css.matchAll(/@utility (type-[\w-]+)/g)].map(([, role]) => role);
		expect(roles.filter((role) => !page.includes(`'${role}'`))).toEqual([]);
	});
});

describe('shape names a radius role', () => {
	test('no stock radius step; circles use rounded-full, objects an exact size', () => {
		expect(offenders(utility('rounded(?:-[tblrsexy]{1,2})?-(?:xs|sm|md|lg|xl|[2-4]xl)'))).toEqual(
			[],
		);
	});
});

describe('metadata colours mirror the tokens', () => {
	test('viewport themeColor matches neutral-50 and neutral-950', () => {
		const css = readFileSync('src/app/globals.css', 'utf8');
		const layout = readFileSync('src/app/layout.tsx', 'utf8');
		for (const step of ['50', '950']) {
			const token = css.match(new RegExp(`--palette-neutral-${step}: (oklch\\([^)]+\\))`))?.[1];
			expect(token).toBeDefined();
			expect(layout).toContain(`color: '${token}'`);
		}
	});
});
