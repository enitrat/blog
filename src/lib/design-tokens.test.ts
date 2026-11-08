import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { Glob } from 'bun';

const files = [...new Glob('src/**/*.{ts,tsx}').scanSync('.')];

function offenders(pattern: RegExp) {
	return files.flatMap((file) =>
		readFileSync(file, 'utf8')
			.split('\n')
			.flatMap((line, i) => (pattern.test(line) ? [`${file}:${i + 1}: ${line.trim()}`] : [])),
	);
}

const COLOR_UTILITY =
	'(?:bg|text|border|ring|ring-offset|outline|fill|stroke|from|via|to|decoration|divide|shadow|placeholder|caret|accent)';

describe('components name roles, not ramps or themes', () => {
	test('no primitive ramp step in a class', () => {
		const ramp = new RegExp(
			`\\b${COLOR_UTILITY}-(?:paper|oxblood|bottle|brass|walnut|gray|iron|brand|zinc|neutral|slate|stone)-\\d`,
		);
		expect(offenders(ramp)).toEqual([]);
	});

	test('no dark: colour variant, since roles swap with the theme', () => {
		const darkColor = new RegExp(`dark:(?:[\\w-]+:)*${COLOR_UTILITY}-(?!\\(--shiki)`);
		expect(offenders(darkColor)).toEqual([]);
	});
});

describe('metadata colours mirror the tokens', () => {
	test('viewport themeColor matches paper-50 and paper-950', () => {
		const css = readFileSync('src/app/globals.css', 'utf8');
		const layout = readFileSync('src/app/layout.tsx', 'utf8');
		for (const step of ['50', '950']) {
			const token = css.match(new RegExp(`--color-paper-${step}: (oklch\\([^)]+\\))`))?.[1];
			expect(token).toBeDefined();
			expect(layout).toContain(`color: '${token}'`);
		}
	});
});
