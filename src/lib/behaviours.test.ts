import { describe, expect, test } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import sharp from 'sharp';
import { offenders } from './source-lines';

/**
 * Site behaviours are decided once, in globals.css, Providers and src/lib, and components inherit
 * them. Each test here fails on the local workaround that the global rule replaced, so a new
 * component cannot quietly re-decide one.
 */

/** Dialog popups take focus programmatically and are not themselves a control. */
const FOCUS_CONTAINERS = ['src/components/ui/dialog.tsx', 'src/components/places/lightbox.tsx'];

describe('focus is global (globals.css `:focus-visible`)', () => {
	test('no component draws or removes its own focus ring', () => {
		const local = /focus(?:-visible)?:ring|\boutline-none\b/;
		expect(offenders(local, (file) => FOCUS_CONTAINERS.includes(file))).toEqual([]);
	});
});

describe('reduced motion is global (globals.css and <MotionConfig reducedMotion="user">)', () => {
	test('no motion-reduce or motion-safe variant; `motion-only` names an element that only moves', () => {
		expect(offenders(/\bmotion-(?:reduce|safe):/)).toEqual([]);
	});

	test('no transition zeroed by hand for reduced motion', () => {
		expect(offenders(/\?\s*\{\s*duration:\s*0\s*\}/)).toEqual([]);
	});
});

describe('hit areas and stretched links are utilities', () => {
	test('no hand-sized ::after; use `hit-area` (44px minimum) or `link-cover`', () => {
		expect(offenders(/\bafter:(?:-?inset|absolute)/)).toEqual([]);
	});
});

describe('formats live in src/lib/format.ts', () => {
	test('no date or number formatter outside it', () => {
		const local = /Intl\.(?:DateTimeFormat|NumberFormat)|toLocale(?:Date)?String/;
		expect(offenders(local, (file) => file === 'src/lib/format.ts')).toEqual([]);
	});
});

describe('home tiles size by their container', () => {
	// A tile's width comes from the bento, which the sidebar narrows: the window cannot say how
	// much room a tile has, so tile contents use container queries (`@[..]/name:`), never `sm:`/`lg:`.
	test('no viewport breakpoint inside a tile', () => {
		const viewport = /(?<![\w@/[-])(?:max-)?(?:sm|md|lg|xl|2xl):/;
		const tile = (file: string) =>
			!file.startsWith('src/components/home/') || file.endsWith('home-sidebar.tsx');
		expect(offenders(viewport, tile)).toEqual([]);
	});
});

describe('book covers are portrait', () => {
	// A landscape scan is a wraparound or a flap, and no crop finds the front in it.
	test('every checked-in cover is taller than wide', async () => {
		const dir = 'src/assets/covers';
		const shapes = await Promise.all(
			readdirSync(dir).map(async (name) => {
				const { width = 0, height = 0 } = await sharp(`${dir}/${name}`).metadata();
				return { name, portrait: height > width };
			}),
		);
		expect(shapes.filter((s) => !s.portrait).map((s) => s.name)).toEqual([]);
	});
});

describe('the favicon is the wordmark\u2019s M', () => {
	// A static file cannot import the component, so this keeps the copy honest.
	const favicon = readFileSync('public/favicon.svg', 'utf8');
	const letters = readFileSync('src/components/site/wordmark.tsx', 'utf8').match(
		/const LETTERS =\s*'([^']+)'/,
	)?.[1];
	const css = readFileSync('src/app/globals.css', 'utf8');
	const step = (name: string) =>
		css.match(new RegExp(`--palette-${name}: (oklch\\([^)]+\\))`))?.[1];

	test('it draws the M glyph of the wordmark', () => {
		expect(letters).toBeDefined();
		const glyph = letters?.slice(0, letters.indexOf('Z') + 1) ?? '';
		expect(favicon).toContain(`d="${glyph}"`);
	});

	test('it paints the sticker roles: neutral-0 fill, accent-600 outline, accent-800 shadow', () => {
		const roles = ['neutral-0', 'accent-600', 'accent-800'].map(step);
		expect(roles.every(Boolean)).toBe(true);
		// Every colour the favicon paints is one of the three, and each of the three is used.
		const painted = new Set(favicon.match(/oklch\([^)]+\)/g));
		expect([...painted].sort()).toEqual([...new Set(roles)].sort() as string[]);
		expect(css).toContain('--sticker-fill: var(--palette-neutral-0)');
		expect(css).toContain('--sticker-outline: var(--palette-accent-600)');
		expect(css).toContain('--sticker-shadow: var(--palette-accent-800)');
	});
});
