/**
 * Generates src/styles/tokens.css from the DESIGN.md YAML frontmatter, the
 * single source of truth for reference tokens. Run with --check to fail
 * (exit 1) when the committed file differs from what the frontmatter
 * produces — the build runs that mode so docs and CSS cannot drift apart.
 *
 * Emits reference tokens only (colors, font stacks, type roles, spacing).
 * Intent-level aliases (--home-*) stay hand-written in site.css and must
 * point at these.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const designPath = join(root, 'DESIGN.md');
const tokensPath = join(root, 'src/styles/tokens.css');

const design = readFileSync(designPath, 'utf8');
const match = design.match(/^---\n([\s\S]*?)\n---\n/);
if (!match) {
	console.error('generate-tokens: no YAML frontmatter found in DESIGN.md');
	process.exit(1);
}
const spec = parse(match[1]);

const lines = [
	'/* GENERATED from DESIGN.md frontmatter — do not edit by hand.',
	' * Regenerate with: pnpm tokens. The build fails if this file is stale. */',
	':root {',
];

for (const [name, value] of Object.entries(spec.colors ?? {})) {
	lines.push(`\t--color-${name}: ${value};`);
}

// One variable per unique font stack, named by its generic family.
const stacks = new Map();
for (const role of Object.values(spec.typography ?? {})) {
	const stack = role.fontFamily;
	if (!stack || stacks.has(stack)) continue;
	const generic =
		/serif$/.test(stack) && !/sans-serif$/.test(stack)
			? 'serif'
			: /monospace$/.test(stack)
				? 'mono'
				: 'sans';
	stacks.set(stack, `--font-${generic}`);
}
for (const [stack, varName] of stacks) {
	lines.push(
		`\t${varName}: ${stack.replace(/([A-Za-z][\w ]+ [\w]+)(?=,)/g, '"$1"').replace(/^([A-Za-z][\w ]+ [\w]+)/, '"$1"')};`,
	);
}

const propMap = {
	fontSize: 'size',
	fontWeight: 'weight',
	lineHeight: 'leading',
	letterSpacing: 'tracking',
};
for (const [role, props] of Object.entries(spec.typography ?? {})) {
	for (const [prop, suffix] of Object.entries(propMap)) {
		if (props[prop] !== undefined) {
			lines.push(`\t--type-${role}-${suffix}: ${props[prop]};`);
		}
	}
	if (props.fontFamily) {
		lines.push(`\t--type-${role}-family: var(${stacks.get(props.fontFamily)});`);
	}
}

for (const [name, value] of Object.entries(spec.spacing ?? {})) {
	lines.push(`\t--space-${name}: ${value};`);
}

lines.push('}', '');
const output = lines.join('\n');

if (process.argv.includes('--check')) {
	let current = '';
	try {
		current = readFileSync(tokensPath, 'utf8');
	} catch {
		// missing file is stale by definition
	}
	if (current !== output) {
		console.error(
			'generate-tokens: src/styles/tokens.css is stale relative to DESIGN.md. Run: pnpm tokens',
		);
		process.exit(1);
	}
	console.log('generate-tokens: tokens.css matches DESIGN.md');
} else {
	writeFileSync(tokensPath, output);
	console.log(`generate-tokens: wrote ${tokensPath}`);
}
