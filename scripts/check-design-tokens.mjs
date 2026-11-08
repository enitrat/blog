/**
 * `src/ds/tokens/` is the one source of every design value. DESIGN.md's frontmatter describes the
 * same values for the impeccable skill; this check fails the build when the two disagree, so the
 * document cannot drift from what ships. Fix a failure in whichever side is wrong (DESIGN.md only
 * through the impeccable skill).
 */
import { readFileSync } from 'node:fs';
import { parse } from 'yaml';

const read = (f) => readFileSync(`src/ds/tokens/${f}`, 'utf8');
const css = ['primitives.css', 'typography.css', 'radius.css', 'layout.css', 'motion.css']
	.map(read)
	.join('\n');
/** The first declared value of a custom property, whitespace collapsed. */
const declared = (name) => {
	const m = css.match(new RegExp(`--${name}:\\s*([^;]+);`));
	return m?.[1].replace(/\s+/g, ' ').trim();
};
const norm = (v) => String(v).replace(/\s+/g, ' ').trim();

const spec = parse(readFileSync('DESIGN.md', 'utf8').match(/^---\n([\s\S]*?)\n---\n/)[1]);
const pairs = [
	...Object.entries(spec.colors ?? {}).map(([k, v]) => [k, v]),
	...Object.entries(spec.fontSize ?? {}).map(([k, v]) => [`text-${k}`, v]),
	...Object.entries(spec.leading ?? {}).map(([k, v]) => [`leading-${k}`, v]),
	...Object.entries(spec.weight ?? {}).map(([k, v]) => [`font-weight-${k}`, v]),
	...Object.entries(spec.tracking ?? {}).map(([k, v]) => [`tracking-${k}`, v]),
	...Object.entries(spec.rounded ?? {}).map(([k, v]) => [
		`radius-${k}`,
		v === '999px' ? '999px' : v,
	]),
	...Object.entries(spec.motion?.duration ?? {}).map(([k, v]) => [`duration-${k}`, v]),
	...Object.entries(spec.motion?.ease ?? {}).map(([k, v]) => [`ease-${k}`, v]),
	...Object.entries(spec.elevation ?? {}).map(([k, v]) => [
		k.endsWith('-dark') ? `shadow-${k}` : `shadow-${k}-light`,
		v,
	]),
];
const LAYOUT = {
	'page-gutter': 'page-gutter',
	'bench-gap': 'gap-bench',
	'tile-pad': 'pad-tile',
	'page-max': 'page-max',
	measure: 'measure',
};
for (const [k, v] of Object.entries(spec.layout ?? {})) pairs.push([LAYOUT[k] ?? k, v]);

const errors = [];
for (const [name, value] of pairs) {
	const got = declared(name);
	if (got === undefined) errors.push(`--${name} is in DESIGN.md but not in src/ds/tokens`);
	else if (norm(got) !== norm(value))
		errors.push(`--${name}: DESIGN.md says ${value}, src/ds/tokens says ${got}`);
}
// The spacing scale is `--spacing` (0.25rem) times the step.
for (const [k, v] of Object.entries(spec.space ?? {})) {
	if (norm(v) !== `${Number(k) * 0.25}rem`)
		errors.push(`space ${k} = ${v} is off the 0.25rem scale`);
}
if (errors.length) {
	console.error(
		`check-design-tokens: DESIGN.md and src/ds/tokens disagree:\n  ${errors.join('\n  ')}`,
	);
	process.exit(1);
}
console.log(`check-design-tokens: ${pairs.length} values match DESIGN.md`);
