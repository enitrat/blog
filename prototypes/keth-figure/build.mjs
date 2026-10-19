// node build.mjs [--light] : writes keth-figure.html (dark) or keth-figure-light.html next to this file.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildFigure, controlHtml, KEYS } from './figure.mjs';
import { kitScript } from './inline-kit.mjs';
import * as k from './kit/iso-kit.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const light = process.argv.includes('--light');
const { svg, P } = buildFigure();
const read = (f) =>
	readFileSync(join(HERE, f), 'utf8')
		.replace(/^import .*$/gm, '')
		.replace(/^export /gm, '');
const script = `${kitScript()}\n${read('model.mjs')}\n${read('live.mjs')}
const root = document.querySelector('.keth');
mount(root, { P: ${JSON.stringify(P)}, translateAlong, motion: () => !matchMedia('(prefers-reduced-motion: reduce)').matches,
  say: (text) => { document.querySelector('[data-readout]').textContent = text; } });`;

const body = k.plateHtml({
	fig: 'Fig 1',
	title: 'Keth',
	hint: 'Slide to resize the block',
	readout: '30 txs · 3 body · block',
	keys: KEYS,
	caption:
		'One Prague block becomes prover inputs and is cut into an init segment, body chunks of ten transactions and a teardown. A fixed set of sixteen mpt_diff chunks, one per branch of the state trie, sits beside them. Each chunk is run and stamped, an aggregator checks that the stamps chain, and one proof is made and verified.',
	body: `<div class="keth"><div class="keth-stage">${svg}</div>${controlHtml()}</div>`,
});
const css = `<style>
.keth-control{display:grid;gap:6px;margin:10px auto 0;max-width:420px}
.keth-label{font:10.5px var(--anatomy-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--anatomy-muted)}
.keth-range input{width:100%;accent-color:var(--anatomy-strong);margin:0}
.keth-ticks{display:flex;justify-content:space-between;font:10.5px var(--anatomy-mono);color:var(--anatomy-muted);padding:0 6px}
</style>`;
const out = k
	.pageHtml({ title: 'Keth figure', theme: light ? 'light' : 'dark', body, script, width: 780 })
	.replace('</head>', `${css}</head>`);
const name = light ? 'keth-figure-light.html' : 'keth-figure.html';
writeFileSync(join(HERE, name), out);
console.log('wrote', name, `${Math.round(out.length / 1024)} KB`);
