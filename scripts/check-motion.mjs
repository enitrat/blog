/**
 * Motion-quality checks for the home bench: decks, tilt, magnetic, glide, linked tiles, settle.
 *
 *   bun run motion:check                 every installed engine and viewport
 *   bun run motion:check chromium        one engine (chromium | webkit | firefox | all)
 *
 * Measures a production build by default: `astro build`, then serves dist/ statically, so Vite's
 * client and the dev toolbar are not on the page. `--dev` (or DEV_URL) uses a dev server instead
 * (a running one at http://localhost:4321, else its own); dev-only scripts are then listed but not
 * counted. MOTION_NOBUILD=1 reuses the existing dist/.
 * MOTION_ONLY=5 runs one check by its number; MOTION_TRACE=1 writes a Chromium trace of the deck scenarios. Writes a video per viewport and a JSON report per engine to MOTION_OUT
 * (default /tmp/motion-check, like room:check's /tmp/room-check).
 *
 * Checks (each PASS / FAIL / SKIP with numbers):
 *   1 interruption continuity   2 zone stability     3 layout reads per pointermove
 *   4 animated properties       5 frame budget       6 reduced motion       7 touch inertness
 *
 * A "frame" is one requestAnimationFrame callback. Steps are normalised to 60Hz so a slow frame
 * does not look like a jump.
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { chromium, firefox, webkit } from 'playwright';
import { serve } from './dev-server.mjs';

// ---- thresholds -------------------------------------------------------------------------------
const JUMP_RATIO = 1.5; // a step this many times the larger of its two neighbouring steps (60Hz-normalised)...
const JUMP_MIN_PX = 6; // ...and at least this long is a snap
const JUMP_MAX_DEG = 4; // rotation per frame above this is a snap
const MOVING_EPS = 0.01; // px or deg per frame that counts as moving
const DECK_HALF_MS = 120; // roughly halfway through SPRING_LAYOUT
const TILT_HALF_MS = 100; // roughly halfway through SPRING_MOUSE
const GLIDE_HALF_MS = 70; // roughly halfway through SPRING_GLIDE
const REST_MS = 1100; // let every spring come to rest between scenarios
const AFTER_MS = 900; // keep sampling this long after the last action
const ZONE_STEP_PX = 4;
const READS_MAX_PER_EVENT = 1;
const POINTER_MOVES = 60;
const LOAF_MAX_MS = 50;
const ANIMATED_MIN_FRAMES = 3; // a property changing in this many consecutive frames is animated, fewer is a discrete switch
const REDUCED_MAX_CHANGED_FRAMES = 1;
const TOUCH_MAX_DRIFT = 0.5; // px or deg a touch page may move on its own
const ALLOWED_PROPS = new Set(['transform', 'opacity', 'box-shadow', 'clip-path']);
// Colour state is allowed by the README ("--duration-fast: state (colour, border, press)"):
// paint-only, so tolerated and listed rather than failed. Layout, filter etc. still fail.
const TOLERATED_PROPS = new Set([
	'color',
	'background-color',
	'border-color',
	'border-top-color',
	'border-right-color',
	'border-bottom-color',
	'border-left-color',
	'outline-color',
	'text-decoration-color',
	'fill',
	'stroke',
]);
const FRAME_MS = 1000 / 60;
const MIN_FRAME_MS = 4; // frames closer than this are one display frame
const MAX_MEDIAN_FRAME_MS = 40; // slower than 25fps and frame-to-frame numbers mean nothing
const fpsSeen = [];
let reloads = 0;
const DEV = process.argv.includes('--dev');
const DEFAULT_URL = 'http://localhost:4321';
const OUT = process.env.MOTION_OUT ?? '/tmp/motion-check';

const ENGINES = { chromium, webkit, firefox };
const VIEWPORTS = [
	{ name: 'desktop', width: 1440, height: 900 },
	{ name: 'laptop', width: 1024, height: 768 },
	{ name: 'phone', width: 390, height: 844, hasTouch: true, isMobile: true },
];
const SETTLE_TILES = { name: 'settle', sel: '.bench[data-settle] > *', i: 0, sub: 'all' };

const arg = process.argv.slice(2).find((a) => a in ENGINES || a === 'all') ?? 'all';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const r2 = (x) => Math.round(x * 100) / 100;
const median = (a) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : 0);
const installed = (name) => {
	try {
		return existsSync(ENGINES[name].executablePath());
	} catch {
		return false;
	}
};

// ---- in-page instrument (runs before every page script) ----------------------------------------
function instrument() {
	const m = {};
	window.__m = m;
	// The checker's own reads go through these, captured before countReads() patches the page's
	// copies, so the instrument never counts itself.
	const nativeRect = Element.prototype.getBoundingClientRect;
	const nativeStyle = window.getComputedStyle;
	// The room's WebGL scene drags headless Chromium to ~4.5fps, which makes frame-to-frame numbers
	// meaningless. No WebGL context means the room stays on its poster; nothing here tests the room.
	const getContext = HTMLCanvasElement.prototype.getContext;
	HTMLCanvasElement.prototype.getContext = function (type, ...a) {
		return /webgl/i.test(type) ? null : getContext.call(this, type, ...a);
	};
	const style = document.createElement('style');
	style.textContent = 'astro-dev-toolbar{display:none!important}';
	if (document.documentElement) document.documentElement.append(style);
	else document.addEventListener('DOMContentLoaded', () => document.documentElement.append(style));

	const label = (el) =>
		`${el.tagName.toLowerCase()}.${String(el.className?.baseVal ?? el.className).split(' ')[0]}`;
	const rect = (el) => {
		const r = nativeRect.call(el);
		return { l: r.left, t: r.top, r: r.right, b: r.bottom, w: r.width, h: r.height };
	};
	/** spec {name, sel, i, sub: self|child|cards|all} -> [{el, label}] */
	const find = (s) => {
		const all = [...document.querySelectorAll(s.sel)];
		const el = all[s.i ?? 0];
		if (s.sub === 'all') return all.slice(0, 6).map((e, k) => ({ el: e, label: `${s.name}#${k}` }));
		if (!el) return [];
		if (s.sub === 'cards')
			return [...el.querySelectorAll('[data-deck-card]')].map((e, k) => ({
				el: e,
				label: `${s.name} card#${k}`,
			}));
		if (s.sub === 'child') return [{ el: el.firstElementChild || el, label: s.name }];
		return [{ el, label: s.name }];
	};
	const deg = (r) => (r * 180) / Math.PI;
	const vec = (el, glide) => {
		if (glide) {
			const g = (k) => Number.parseFloat(el.style.getPropertyValue(k)) || 0;
			return [g('--gt'), g('--gr'), g('--gb'), g('--gl'), el.hasAttribute('data-glide-on') ? 1 : 0];
		}
		const t = nativeStyle.call(window, el).transform;
		const q = t === 'none' ? new DOMMatrix() : new DOMMatrix(t);
		return [
			q.m41,
			q.m42,
			deg(Math.atan2(q.m12, q.m11)),
			deg(Math.atan2(q.m23, q.m33)),
			deg(Math.atan2(q.m13, q.m33)),
			Math.hypot(q.m11, q.m12),
		];
	};

	// Sampler: every animation frame, one vector per element. Elements are re-resolved each frame
	// so it can start before the page has parsed (settle).
	let run = null;
	m.start = (specs, glide = false) => {
		const r = { on: true, frames: [], marks: [], labels: [] };
		run = r;
		// Sample after every rAF callback of the frame has run (a message task), stamped with the
		// frame's own timestamp: otherwise the order of other scripts' callbacks adds jitter.
		const ch = new MessageChannel();
		let ts = 0;
		ch.port1.onmessage = () => {
			if (!r.on) return;
			// no specs: frame timing only, no style reads (the frame budget check must not add its own)
			if (!specs.length) r.frames.push({ t: ts, v: [] });
			const found = specs.flatMap(find);
			if (found.length) {
				r.labels = found.map((f) => f.label);
				r.frames.push({ t: ts, v: found.map((f) => vec(f.el, glide)) });
			}
			requestAnimationFrame(loop);
		};
		function loop(now) {
			ts = now;
			ch.port2.postMessage(0);
		}
		requestAnimationFrame(loop);
	};
	m.mark = (name) => run?.marks.push({ t: performance.now(), name });
	m.stop = () => {
		const r = run;
		run = null;
		if (!r) return { frames: [], marks: [], labels: [] };
		r.on = false;
		return r;
	};

	m.geo = (s) => {
		const el = document.querySelectorAll(s.sel)[s.i ?? 0];
		if (!el) return null;
		el.scrollIntoView({ block: 'center', behavior: 'instant' });
		const box = s.parent ? el.parentElement : el;
		const host = el.closest('[data-deck-host]') || el;
		const sub = s.sub === 'child' ? el.firstElementChild || el : el;
		const g = {
			el: rect(el),
			box: rect(box),
			host: rect(host),
			item: rect(sub),
			n: el.querySelectorAll('[data-deck-card]').length,
			spread: Number(el.dataset.spread),
			items: [...el.querySelectorAll('[data-glide-item]')].map(rect),
			vw: innerWidth,
			vh: innerHeight,
		};
		return g.el.w > 0 && g.el.h > 0 ? g : null;
	};
	m.count = () => {
		const n = (s) => document.querySelectorAll(s).length;
		return {
			deck: n('[data-deck]'),
			tilt: n('[data-tilt]'),
			magnetic: n('[data-magnetic]'),
			glide: n('[data-glide]'),
			tile: n('.tile[data-linked]'),
		};
	};
	m.state = (s) => {
		const el = document.querySelectorAll(s.sel)[s.i ?? 0];
		return {
			open: el?.hasAttribute('data-open'),
			active: el?.dataset.active,
			tilt: el?.hasAttribute('data-tilt-active'),
		};
	};
	m.targets = () => {
		const sels = '[data-deck],[data-tilt],[data-magnetic],[data-glide],.tile[data-linked]';
		return [...document.querySelectorAll(sels)]
			.map(rect)
			.filter((r) => r.w > 0 && r.b > 0 && r.t < innerHeight);
	};
	m.scrollTo = (y) => scrollTo({ top: y, behavior: 'instant' });
	m.scrollMax = () => document.documentElement.scrollHeight - innerHeight;

	// Animated-property watcher: running animations plus inline-style mutations, per frame.
	m.watch = () => {
		const recs = new Map();
		let frame = 0;
		let on = true;
		const seen = { min: Number.POSITIVE_INFINITY, max: 0 };
		const rec = (prop) => {
			if (!recs.has(prop)) recs.set(prop, { anim: new Set(), style: new Set(), who: new Set() });
			return recs.get(prop);
		};
		const kebab = (s) => s.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
		const SKIP = new Set(['offset', 'easing', 'composite', 'computed-offset']);
		const tmp = document.createElement('div');
		const parse = (s) => {
			tmp.style.cssText = s || '';
			const o = {};
			for (let i = 0; i < tmp.style.length; i++)
				o[tmp.style[i]] = tmp.style.getPropertyValue(tmp.style[i]);
			return o;
		};
		const mo = new MutationObserver((list) => {
			for (const mu of list) {
				const before = parse(mu.oldValue);
				const after = parse(mu.target.getAttribute('style'));
				for (const k of new Set([...Object.keys(before), ...Object.keys(after)])) {
					if (before[k] === after[k]) continue;
					const x = rec(k);
					x.style.add(frame);
					x.who.add(label(mu.target));
				}
			}
		});
		mo.observe(document, {
			attributes: true,
			attributeFilter: ['style'],
			attributeOldValue: true,
			subtree: true,
		});
		const loop = () => {
			if (!on) return;
			frame++;
			for (const a of document.getAnimations()) {
				const e = a.effect;
				const who = e?.target ? label(e.target) + (e.pseudoElement ?? '') : '?';
				let props = [];
				if (a.transitionProperty) props = [a.transitionProperty];
				else if (e?.getKeyframes)
					props = [...new Set(e.getKeyframes().flatMap((k) => Object.keys(k)))]
						.map(kebab)
						.filter((p) => !SKIP.has(p));
				for (const p of props) {
					const x = rec(p);
					x.anim.add(frame);
					x.who.add(`${who}${a.animationName ? ` (${a.animationName})` : ''}`);
				}
				if (a.animationName?.startsWith('bench-')) {
					seen.min = Math.min(seen.min, performance.now());
					seen.max = Math.max(seen.max, performance.now());
				}
			}
			requestAnimationFrame(loop);
		};
		requestAnimationFrame(loop);
		// longest run of consecutive frames: a z-index flip per event is a switch, not an animation
		const longest = (set) => {
			let best = 0;
			let cur = 0;
			let prev = -2;
			for (const f of [...set].sort((a, b) => a - b)) {
				cur = f === prev + 1 ? cur + 1 : 1;
				prev = f;
				best = Math.max(best, cur);
			}
			return best;
		};
		m.unwatch = () => {
			on = false;
			mo.disconnect();
			return {
				settle: seen,
				props: [...recs].map(([prop, x]) => ({
					prop,
					anim: longest(x.anim),
					style: longest(x.style),
					who: [...x.who].slice(0, 4),
				})),
			};
		};
	};

	// Long animation frames, buffered from the start of the page.
	m.loaf = [];
	try {
		m.po = new PerformanceObserver((l) => {
			for (const e of l.getEntries())
				m.loaf.push({
					s: e.startTime,
					d: e.duration,
					render: e.renderStart - e.startTime,
					styleLayout: e.startTime + e.duration - e.styleAndLayoutStart,
					blocking: e.blockingDuration,
					scripts: e.scripts.map((x) => ({
						who: `${x.invokerType}:${x.invoker}`,
						src: `${x.sourceURL.replace(/^https?:\/\/[^/]+/, '')}:${x.sourceFunctionName}`,
						ms: Math.round(x.duration),
						forced: Math.round(x.forcedStyleAndLayoutDuration),
					})),
				});
		});
		m.po.observe({ type: 'long-animation-frame', buffered: true });
		m.loafSupported = true;
	} catch {
		m.loafSupported = false;
	}
	m.takeLoaf = () => {
		for (const e of m.po?.takeRecords() ?? [])
			m.loaf.push({
				s: e.startTime,
				d: e.duration,
				render: e.renderStart - e.startTime,
				styleLayout: e.startTime + e.duration - e.styleAndLayoutStart,
				blocking: e.blockingDuration,
				scripts: e.scripts.map((x) => ({
					who: `${x.invokerType}:${x.invoker}`,
					src: `${x.sourceURL.replace(/^https?:\/\/[^/]+/, '')}:${x.sourceFunctionName}`,
					ms: Math.round(x.duration),
					forced: Math.round(x.forcedStyleAndLayoutDuration),
				})),
			});
		return { supported: m.loafSupported, entries: m.loaf, now: performance.now() };
	};

	// Layout-read counters, installed on demand so page load is not counted.
	m.countReads = () => {
		const c = { total: 0, by: {}, sites: {}, dev: {}, moves: 0 };
		m.reads = c;
		// Vite's client and Astro's dev toolbar exist only under `astro dev`; they are listed, not counted.
		const DEV = /\/@vite\/|\/@id\/|dev-toolbar|\/@fs\//;
		const hit = function __hit(name) {
			// up to three frames, each as pathname:line, skipping the wrappers above
			const frames = (new Error().stack || '')
				.split('\n')
				.filter((l) => /:\d+:\d+\)?$/.test(l) && !l.includes('__hit') && !l.includes('__w'))
				.slice(0, 3)
				.map((l) => {
					const f = l.match(/((?:https?|file):\/\/[^\s()]+?:\d+):(\d+)\)?$/);
					return f
						? `${f[1].replace(/^https?:\/\/[^/]+/, '').replace(/\?[^:]*(?=:\d+$)/, '')}:${f[2]}`
						: l.trim();
				});
			const site = frames.join(' < ') || '?';
			const key = `${name} @ ${site}`;
			if (DEV.test(frames[0] ?? '')) {
				c.dev[key] = (c.dev[key] || 0) + 1;
				return;
			}
			c.total++;
			c.by[name] = (c.by[name] || 0) + 1;
			c.sites[key] = (c.sites[key] || 0) + 1;
		};
		Element.prototype.getBoundingClientRect = function __w(...a) {
			hit('getBoundingClientRect');
			return nativeRect.apply(this, a);
		};
		for (const k of ['offsetWidth', 'offsetHeight']) {
			const d = Object.getOwnPropertyDescriptor(HTMLElement.prototype, k);
			Object.defineProperty(HTMLElement.prototype, k, {
				...d,
				get: function __w() {
					hit(k);
					return d.get.call(this);
				},
			});
		}
		window.getComputedStyle = function __w(...a) {
			hit('getComputedStyle');
			return nativeStyle.apply(this, a);
		};
		document.addEventListener('pointermove', () => c.moves++, true);
	};
	m.readsDone = () => m.reads;

	// Touch: count state attributes appearing.
	m.attrs = () => {
		let n = 0;
		const names = [];
		new MutationObserver((l) => {
			for (const mu of l) {
				n++;
				names.push(`${mu.attributeName} on ${label(mu.target)}`);
			}
		}).observe(document, {
			attributes: true,
			attributeFilter: ['data-open', 'data-active', 'data-tilt-active'],
			subtree: true,
		});
		m.attrCount = () => ({ n, names: names.slice(0, 5) });
	};
	m.scrollTour = async () => {
		const max = document.documentElement.scrollHeight - innerHeight;
		for (let y = 0; y <= max; y += 40) {
			scrollTo({ top: y, behavior: 'instant' });
			await new Promise((r) => requestAnimationFrame(r));
		}
		for (let y = max; y >= 0; y -= 80) {
			scrollTo({ top: y, behavior: 'instant' });
			await new Promise((r) => requestAnimationFrame(r));
		}
	};
}

// ---- page + pointer helpers ---------------------------------------------------------------------
async function openPage(ctx, base, { autostart, watch } = {}) {
	const page = await ctx.newPage();
	let live = false;
	// Saving a source file makes the dev server reload the page under the check: count it and rerun.
	page.on('framenavigated', (f) => {
		if (live && f === page.mainFrame()) reloads++;
	});
	if (autostart) await page.addInitScript((s) => window.__m.start(s), autostart);
	if (watch) await page.addInitScript(() => window.__m.watch());
	await page.goto(base, { waitUntil: 'load' });
	await page.waitForSelector('[data-glide-ready]', { timeout: 15000 });
	await sleep(autostart || watch ? 1500 : 900);
	live = true;
	return page;
}
const call = (page, fn, ...args) => page.evaluate(([f, a]) => window.__m[f](...a), [fn, args]);

function pointer(page) {
	let x = 0;
	let y = 0;
	return {
		async to(nx, ny, ms = 0) {
			const steps = ms ? Math.max(1, Math.round(ms / FRAME_MS)) : 1;
			const [x0, y0] = [x, y];
			for (let i = 1; i <= steps; i++) {
				await page.mouse.move(x0 + ((nx - x0) * i) / steps, y0 + ((ny - y0) * i) / steps);
				if (ms) await sleep(FRAME_MS);
			}
			[x, y] = [nx, ny];
		},
	};
}
const cx = (r) => r.l + r.w / 2;
const cy = (r) => r.t + r.h / 2;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
/** A point just outside rect `r` that is still on screen. */
function outside(r, g, gap = 24) {
	const tries = [
		[cx(r), r.t - gap],
		[cx(r), r.b + gap],
		[r.l - gap, cy(r)],
		[r.r + gap, cy(r)],
	];
	const ok = tries.find(([x, y]) => x > 2 && y > 2 && x < g.vw - 2 && y < g.vh - 2);
	return ok ?? [clamp(r.l - gap, 2, g.vw - 2), clamp(r.t - gap, 2, g.vh - 2)];
}
const footprint = (g) => {
	const w = g.el.w * (1 + g.spread * (g.n - 1));
	const l = g.el.l - (g.el.w * g.spread * (g.n - 1)) / 2;
	return { l, w, r: l + w };
};

/** Run `fn(mark, mouse)` under the sampler and return what it saw. */
async function sampled(page, specs, fn, { glide = false, after = AFTER_MS } = {}) {
	await call(page, 'start', specs, glide);
	const mouse = pointer(page);
	await fn((name) => call(page, 'mark', name), mouse);
	await sleep(after);
	const run = await call(page, 'stop');
	// two rAF callbacks in one display frame (dt ~1ms) would read as a huge normalised step
	const kept = [];
	for (const f of run.frames) if (!kept.length || f.t - kept.at(-1).t >= MIN_FRAME_MS) kept.push(f);
	run.frames = kept;
	const dts = run.frames.slice(1).map((f, i) => f.t - run.frames[i].t);
	run.medianDt = median(dts);
	if (dts.length) fpsSeen.push(1000 / run.medianDt);
	if (run.medianDt > MAX_MEDIAN_FRAME_MS)
		throw new Error(
			`unmeasurable: median frame ${r2(run.medianDt)}ms (${r2(1000 / run.medianDt)}fps); something is starving the page`,
		);
	return run;
}

// ---- analysis -----------------------------------------------------------------------------------
const phaseAt = (run, t) => run.marks.filter((m) => m.t <= t).at(-1);
function stepsOf(a, b, dt, glide) {
	const k = FRAME_MS / Math.max(dt, 1);
	if (glide) {
		if (!a[4] || !b[4]) return null; // highlight hidden: it appears in place by design
		return { tr: Math.max(...[0, 1, 2, 3].map((i) => Math.abs(b[i] - a[i]))) * k, rot: 0 };
	}
	return {
		tr: Math.hypot(b[0] - a[0], b[1] - a[1]) * k,
		rot: Math.max(Math.abs(b[2] - a[2]), Math.abs(b[3] - a[3]), Math.abs(b[4] - a[4])) * k,
	};
}
/** Per element: median moving step, and every frame whose step is a snap. */
function findJumps(run, glide = false) {
	const jumps = [];
	let worst = 0;
	for (let k = 0; k < (run.frames[0]?.v.length ?? 0); k++) {
		const steps = [];
		for (let f = 1; f < run.frames.length; f++) {
			const [p, q] = [run.frames[f - 1], run.frames[f]];
			if (!p.v[k] || !q.v[k]) continue;
			const s = stepsOf(p.v[k], q.v[k], q.t - p.t, glide);
			steps.push(s && { ...s, f, dt: q.t - p.t });
		}
		for (let i = 0; i < steps.length; i++) {
			const s = steps[i];
			if (!s) continue;
			worst = Math.max(worst, s.tr);
			// Velocity discontinuity: a step well above BOTH neighbours is a snap. A spring that is
			// moving fast, or that starts stiffly, has a bigger neighbour, so it is not flagged.
			const nb = [steps[i - 1], steps[i + 1]].filter(Boolean).map((n) => n.tr);
			const med = nb.length ? Math.max(...nb) : 0;
			const snap = s.tr > JUMP_MIN_PX && s.tr > JUMP_RATIO * med;
			if (!snap && s.rot <= JUMP_MAX_DEG) continue;
			const t = run.frames[s.f].t;
			const ph = phaseAt(run, t);
			jumps.push({
				el: run.labels[k],
				frame: s.f,
				ms: r2(t - run.frames[0].t),
				after: ph ? `${r2(t - ph.t)}ms after '${ph.name}'` : 'before any action',
				px: r2(s.tr),
				median: r2(med),
				deg: r2(s.rot),
				dt: r2(s.dt),
			});
		}
	}
	return { jumps, worst: r2(worst) };
}
const fmtJump = (j) =>
	`${j.el}: ${j.deg > JUMP_MAX_DEG ? `${j.deg}deg` : `${j.px}px (neighbour ${j.median}px)`} in one frame at frame ${j.frame} (t=${j.ms}ms, ${j.after}, dt ${j.dt}ms)`;

/** Frames in [from,to] where anything moved, per element-agnostic. */
function changedFrames(run, from, to, glide = false) {
	const out = [];
	for (let f = 1; f < run.frames.length; f++) {
		const [p, q] = [run.frames[f - 1], run.frames[f]];
		if (q.t < from || q.t > to) continue;
		const moved = q.v.some((v, k) => {
			if (!p.v[k]) return false;
			const n = glide ? 4 : 5;
			return (
				v.slice(0, n).some((x, i) => Math.abs(x - p.v[k][i]) > MOVING_EPS) ||
				(!glide && Math.abs(v[5] - p.v[k][5]) > 1e-4)
			);
		});
		if (moved) out.push(f);
	}
	return out;
}

// ---- targets and scenarios -----------------------------------------------------------------------
/** Everything interactive on the home page, as {kind, name, spec}. */
async function targets(page) {
	const n = await call(page, 'count');
	const out = [];
	for (let i = 0; i < n.deck; i++)
		out.push({ kind: 'deck', name: `deck[${i}]`, spec: { sel: '[data-deck]', i, sub: 'cards' } });
	for (let i = 0; i < n.tilt; i++)
		out.push({ kind: 'tilt', name: `tilt[${i}]`, spec: { sel: '[data-tilt]', i, parent: true } });
	for (let i = 0; i < n.magnetic; i++)
		out.push({
			kind: 'magnetic',
			name: `magnetic[${i}]`,
			spec: { sel: '[data-magnetic]', i, sub: 'child' },
		});
	for (let i = 0; i < n.glide; i++)
		out.push({ kind: 'glide', name: `glide[${i}]`, spec: { sel: '[data-glide]', i } });
	for (let i = 0; i < n.tile; i++)
		out.push({ kind: 'tile', name: `tile[${i}]`, spec: { sel: '.tile[data-linked]', i } });
	return out.map((t) => ({ ...t, spec: { ...t.spec, name: t.name } }));
}

/** Park the pointer off the object and let it come to rest. */
async function park(mouse, g) {
	await mouse.to(...outside(g.host, g));
	await sleep(REST_MS);
}

/** Deck: an opener that hovers once first (ends the ambient deal), then the named interruption. */
async function deckRuns(page, t) {
	const runs = [];
	// One hover first: it ends the ambient `cycle` deal for good, which would otherwise start
	// a spring of its own in the middle of a scenario.
	const first = await call(page, 'geo', t.spec);
	if (!first) return null;
	await sleep(150);
	const prime = pointer(page);
	await prime.to(cx(first.el), cy(first.el));
	await sleep(500);
	await park(prime, first);
	for (const name of ['leave', 'sweep', 'reenter']) {
		const g = await call(page, 'geo', t.spec);
		if (!g) return null;
		await sleep(150);
		const fp = footprint(g);
		const y = cy(g.el);
		const mouse = pointer(page);
		await park(mouse, g);
		runs.push({
			name,
			run: await sampled(page, [t.spec], async (mark) => {
				await mark('enter');
				await mouse.to(g.el.l + 2, y);
				await sleep(DECK_HALF_MS);
				await mark(name);
				if (name === 'leave') await mouse.to(...outside(g.host, g));
				if (name === 'sweep') await mouse.to(fp.r - 4, y);
				if (name === 'reenter') {
					await mouse.to(...outside(g.host, g));
					await sleep(DECK_HALF_MS / 2);
					await mark('reenter');
					await mouse.to(g.el.r - 2, y);
				}
			}),
		});
	}
	return runs;
}

/** Tilt, magnetic and tile: enter, leave mid-flight, re-enter on the far side. */
async function hoverRun(page, t) {
	const g = await call(page, 'geo', t.spec);
	if (!g) return null;
	await sleep(150);
	const margin = t.kind === 'magnetic' ? 16 : 0;
	const b = t.kind === 'tilt' ? g.box : g.el;
	const mouse = pointer(page);
	await mouse.to(
		...outside({ l: b.l - margin, r: b.r + margin, t: b.t - margin, b: b.b + margin }, g, 28),
	);
	await sleep(REST_MS);
	const half = t.kind === 'tile' ? 80 : TILT_HALF_MS;
	return sampled(page, [t.spec], async (mark) => {
		await mark('enter');
		await mouse.to(b.l + b.w * 0.15, b.t + b.h * 0.15);
		await sleep(half);
		await mark('leave');
		await mouse.to(
			...outside({ l: b.l - margin, r: b.r + margin, t: b.t - margin, b: b.b + margin }, g, 28),
		);
		await sleep(half);
		await mark('reenter');
		await mouse.to(b.l + b.w * 0.85, b.t + b.h * 0.85);
	});
}

/** Glide: appear, retarget mid-glide, reverse mid-glide, leave, appear again elsewhere. */
async function glideRun(page, t) {
	const g = await call(page, 'geo', t.spec);
	if (!g || g.items.length < 2) return null;
	await sleep(150);
	const [a, b, c] = [g.items[0], g.items[1], g.items.at(-1)];
	const mouse = pointer(page);
	await mouse.to(...outside(g.el, g, 30));
	await sleep(REST_MS);
	return sampled(
		page,
		[t.spec],
		async (mark) => {
			await mark('appear');
			await mouse.to(cx(a), cy(a));
			await sleep(GLIDE_HALF_MS);
			await mark('retarget');
			await mouse.to(cx(c), cy(c));
			await sleep(GLIDE_HALF_MS);
			await mark('reverse');
			await mouse.to(cx(b), cy(b));
			await sleep(GLIDE_HALF_MS);
			await mark('leave');
			await mouse.to(...outside(g.el, g, 30));
			await sleep(GLIDE_HALF_MS);
			await mark('reappear');
			await mouse.to(cx(c), cy(c));
		},
		{ glide: true },
	);
}

// ---- results -------------------------------------------------------------------------------------
const verdict = (items) => {
	const fail = items.filter((i) => i.status === 'FAIL');
	return fail.length ? 'FAIL' : items.every((i) => i.status === 'SKIP') ? 'SKIP' : 'PASS';
};

// ---- checks --------------------------------------------------------------------------------------
async function check1(ctx, base) {
	const page = await openPage(ctx, base);
	const items = [];
	for (const t of await targets(page)) {
		if (t.kind === 'deck') {
			const runs = await deckRuns(page, t);
			if (!runs) continue;
			for (const { name, run } of runs) {
				const { jumps, worst } = findJumps(run);
				items.push({
					target: `${t.name} ${name}`,
					status: jumps.length ? 'FAIL' : 'PASS',
					worst,
					frames: run.frames.length,
					jumps,
				});
			}
		} else {
			const run = t.kind === 'glide' ? await glideRun(page, t) : await hoverRun(page, t);
			if (!run) continue;
			const { jumps, worst } = findJumps(run, t.kind === 'glide');
			items.push({
				target: t.name,
				status: jumps.length ? 'FAIL' : 'PASS',
				worst,
				frames: run.frames.length,
				jumps,
			});
		}
	}
	await page.close();
	const worst = Math.max(0, ...items.map((i) => i.worst));
	const bad = items.flatMap((i) => i.jumps.map((j) => `${i.target} / ${fmtJump(j)}`));
	return {
		status: verdict(items),
		summary: `${items.length} runs, worst step ${worst}px/frame (jump: >${JUMP_MIN_PX}px and >${JUMP_RATIO}x both neighbours, or >${JUMP_MAX_DEG}deg)`,
		failures: bad,
		items,
	};
}

async function check2(ctx, base) {
	const page = await openPage(ctx, base);
	const items = [];
	for (const t of (await targets(page)).filter((x) => x.kind === 'deck')) {
		const g = await call(page, 'geo', t.spec);
		if (!g) continue;
		await sleep(150);
		const fp = footprint(g);
		const y = cy(g.el);
		const mouse = pointer(page);
		await park(mouse, g);
		await mouse.to(cx(g.el), y);
		const pass = async (from, to) => {
			const seq = [];
			const dir = Math.sign(to - from);
			for (let x = from; dir > 0 ? x <= to : x >= to; x += dir * ZONE_STEP_PX) {
				await mouse.to(x, y);
				const s = await call(page, 'state', t.spec);
				seq.push({ x, open: s.open, active: Number(s.active) });
			}
			return seq;
		};
		const ltr = await pass(fp.l + 1, fp.r - 1);
		const rtl = await pass(fp.r - 1, fp.l + 1);
		const bad = [];
		for (const [name, seq, dir] of [
			['left→right', ltr, 1],
			['right→left', rtl, -1],
		]) {
			let prev = null;
			for (const s of seq) {
				if (!s.open)
					bad.push(
						`${t.name} ${name}: deck closed at x=${r2(s.x)} (pointer left the host mid-sweep)`,
					);
				else {
					if (prev && (s.active - prev.active) * dir < 0)
						bad.push(
							`${t.name} ${name}: active ${prev.active}→${s.active} at x=${r2(s.x)} (non-monotonic)`,
						);
					prev = s;
				}
			}
		}
		// the same x must pick the same zone both ways
		const at = new Map(ltr.map((s) => [r2(s.x), s.active]));
		const hyst = rtl.filter((s) => s.open && at.has(r2(s.x)) && at.get(r2(s.x)) !== s.active);
		if (hyst.length)
			bad.push(
				`${t.name}: ${hyst.length} x positions pick a different zone depending on direction, first at x=${r2(hyst[0].x)}`,
			);
		items.push({
			target: t.name,
			status: bad.length ? 'FAIL' : 'PASS',
			n: g.n,
			steps: ltr.length + rtl.length,
			bad,
		});
	}
	await page.close();
	return {
		status: verdict(items),
		summary: `${items.length} decks, ${items.reduce((a, i) => a + i.steps, 0)} ${ZONE_STEP_PX}px steps, ${items.reduce((a, i) => a + i.bad.length, 0)} anomalies`,
		failures: items.flatMap((i) => i.bad),
		items,
	};
}

async function check3(ctx, base) {
	const page = await openPage(ctx, base);
	await call(page, 'countReads');
	const mouse = pointer(page);
	const max = await call(page, 'scrollMax');
	for (const y of [0, max]) {
		await call(page, 'scrollTo', y);
		await sleep(250);
		// a path through every interactive object in view, interpolated to half the moves
		const pts = (await call(page, 'targets')).map((r) => [cx(r), clamp(cy(r), 4, 890)]);
		const path = [];
		const per = POINTER_MOVES / 2;
		for (let i = 0; i < per; i++) {
			const u = (i / (per - 1)) * (pts.length - 1 || 1);
			const a = pts[Math.floor(u)] ?? [200, 300];
			const b = pts[Math.min(pts.length - 1, Math.ceil(u))] ?? a;
			path.push([a[0] + (b[0] - a[0]) * (u % 1), a[1] + (b[1] - a[1]) * (u % 1)]);
		}
		for (const [x, py] of path) {
			await mouse.to(x, py);
			await sleep(FRAME_MS);
		}
		await sleep(200);
	}
	const c = await call(page, 'readsDone');
	await page.close();
	const per = c.moves ? c.total / c.moves : 0;
	const sites = Object.entries(c.sites)
		.sort((a, b) => b[1] - a[1])
		.slice(0, 5);
	return {
		status: per > READS_MAX_PER_EVENT ? 'FAIL' : 'PASS',
		summary: `${r2(per)} reads per pointermove (${c.total} reads / ${c.moves} events; limit ${READS_MAX_PER_EVENT}); ${
			Object.entries(c.by)
				.map(([k, v]) => `${k} ${v}`)
				.join(', ') || 'none'
		}`,
		failures: per > READS_MAX_PER_EVENT ? sites.map(([k, v]) => `${v}x ${k}`) : [],
		items: [
			{
				target: 'bench',
				reads: c.total,
				moves: c.moves,
				perEvent: r2(per),
				by: c.by,
				topSites: sites,
				devOnly: c.dev,
			},
		],
	};
}

/** Classify the watcher's output; discrete = changed in fewer than ANIMATED_MIN_FRAMES frames. */
function classify(props) {
	const out = { animated: [], tolerated: [], discrete: [], bad: [] };
	for (const p of props) {
		const frames = Math.max(p.anim, p.style);
		const row = `${p.prop} (${p.anim ? `${p.anim} consecutive anim frames` : `${p.style} consecutive style frames`}; ${p.who.join(', ')})`;
		if (p.prop.startsWith('--') || frames < ANIMATED_MIN_FRAMES) {
			(p.prop.startsWith('--') ? out.animated : out.discrete).push(row);
			continue;
		}
		if (ALLOWED_PROPS.has(p.prop)) out.animated.push(row);
		else if (TOLERATED_PROPS.has(p.prop)) out.tolerated.push(row);
		else out.bad.push(row);
	}
	return out;
}

async function check4(ctx, base, touch) {
	const items = [];
	// settle: watcher starts before the page does
	{
		const page = await openPage(ctx, base, { watch: true });
		const res = await call(page, 'unwatch');
		items.push({ target: 'bench settle (page load)', ...classify(res.props) });
		await page.close();
	}
	if (!touch) {
		const page = await openPage(ctx, base);
		const ts = await targets(page);
		for (const t of ts) {
			const g = await call(page, 'geo', t.spec);
			if (!g) continue;
			await sleep(150);
			await call(page, 'watch');
			const mouse = pointer(page);
			const b = t.kind === 'tilt' ? g.box : g.el;
			await mouse.to(...outside(g.host, g));
			await sleep(400);
			const targetsIn =
				t.kind === 'glide' && g.items.length
					? [g.items[0], g.items.at(-1)]
					: [{ l: b.l + 4, t: b.t + 4, w: b.w * 0.8, h: b.h * 0.8 }];
			for (const r of targetsIn) {
				await mouse.to(cx(r), cy(r), 300);
				await sleep(500);
			}
			await mouse.to(...outside(g.host, g));
			await sleep(700);
			items.push({ target: t.name, ...classify((await call(page, 'unwatch')).props) });
		}
		await page.close();
	}
	for (const i of items) i.status = i.bad.length ? 'FAIL' : 'PASS';
	const all = (k) => [...new Set(items.flatMap((i) => i[k].map((r) => r.split(' (')[0])))];
	return {
		status: verdict(items),
		summary: `animated: ${all('animated').join(', ') || 'none'}${all('tolerated').length ? `; colour state tolerated: ${all('tolerated').join(', ')}` : ''}${all('bad').length ? `; DISALLOWED: ${all('bad').join(', ')}` : ''}`,
		failures: items.flatMap((i) => i.bad.map((b) => `${i.target}: ${b}`)),
		items,
	};
}

async function check5(ctx, base, engine) {
	if (engine !== 'chromium')
		return { status: 'SKIP', summary: 'long-animation-frame is chromium only', items: [] };
	const items = [];
	const gaps = (run) => Math.max(0, ...run.frames.slice(1).map((f, i) => f.t - run.frames[i].t));
	const loafIn = (l, from, to) => l.entries.filter((e) => e.s + e.d > from && e.s < to);
	// settle: from document start; only frames that overlap the settle animation itself count
	{
		const page = await openPage(ctx, base, { autostart: [SETTLE_TILES], watch: true });
		const w = await call(page, 'unwatch');
		const run = await call(page, 'stop');
		const l = await call(page, 'takeLoaf');
		const hit = loafIn(l, w.settle.min - 1, w.settle.max + 1);
		items.push({
			target: 'bench settle',
			status: hit.some((e) => e.d > LOAF_MAX_MS) ? 'FAIL' : 'PASS',
			maxGap: r2(gaps(run)),
			loaf: hit.map((e) => ({
				at: r2(e.s),
				ms: r2(e.d),
				render: r2(e.render),
				styleLayout: r2(e.styleLayout),
				scripts: e.scripts,
			})),
			window: [r2(w.settle.min), r2(w.settle.max)],
		});
		await page.close();
	}
	const page = await openPage(ctx, base);
	// MOTION_TRACE=1: a Chromium trace of the deck scenarios, for chrome://tracing or Perfetto.
	const cdp = process.env.MOTION_TRACE ? await ctx.newCDPSession(page) : null;
	const traced = [];
	cdp?.on('Tracing.dataCollected', (e) => traced.push(...e.value));
	await cdp?.send('Tracing.start', {
		traceConfig: {
			includedCategories: [
				'*',
				'disabled-by-default-devtools.timeline',
				'disabled-by-default-v8.cpu_profiler',
			],
		},
	});
	// The first fan of the first deck is measured on its own (`cold`) and gated like every other open:
	// a long frame there is a cold-resource cost (e.g. undecoded back-card images), never noise to exempt.
	/** @type {number[] | null} */
	let cold = null;
	for (const t of (await targets(page)).filter((x) => x.kind === 'deck')) {
		const g = await call(page, 'geo', t.spec);
		if (!g) continue;
		await sleep(150);
		const mouse = pointer(page);
		await park(mouse, g);
		if (!cold) {
			const c0 = (await call(page, 'takeLoaf')).now;
			await mouse.to(g.el.l + 2, cy(g.el), 200);
			await sleep(600);
			await mouse.to(...outside(g.host, g));
			await sleep(REST_MS);
			cold = loafIn(await call(page, 'takeLoaf'), c0, Number.POSITIVE_INFINITY).map((e) => r2(e.d));
		}
		const t0 = (await call(page, 'takeLoaf')).now;
		const run = await sampled(page, [], async (mark) => {
			await mark('open');
			await mouse.to(g.el.l + 2, cy(g.el), 200);
			await sleep(600);
			await mark('close');
			await mouse.to(...outside(g.host, g));
		});
		const hit = loafIn(await call(page, 'takeLoaf'), t0, Number.POSITIVE_INFINITY);
		items.push({
			target: `${t.name} open+close`,
			status: hit.some((e) => e.d > LOAF_MAX_MS) ? 'FAIL' : 'PASS',
			maxGap: r2(gaps(run)),
			loaf: hit.map((e) => ({
				at: r2(e.s),
				ms: r2(e.d),
				render: r2(e.render),
				styleLayout: r2(e.styleLayout),
				scripts: e.scripts,
			})),
		});
	}
	if (cold)
		items.push({
			target: 'first fan of the page (cold)',
			status: cold.some((d) => d > LOAF_MAX_MS) ? 'FAIL' : 'PASS',
			maxGap: 0,
			loaf: cold.map((ms) => ({ ms, at: 'first hover' })),
		});
	if (cdp) {
		const done = new Promise((resolve) => cdp.once('Tracing.tracingComplete', resolve));
		await cdp.send('Tracing.end');
		await done;
		await writeFile(
			`${OUT}/trace-deck-${page.viewportSize().width}.json`,
			JSON.stringify({ traceEvents: traced }),
		);
	}
	await page.close();
	const longest = Math.max(0, ...items.flatMap((i) => i.loaf.map((e) => e.ms)));
	return {
		status: verdict(items),
		summary: `longest LoAF ${longest ? `${longest}ms` : `none over ${LOAF_MAX_MS}ms`} (cold first hover: ${cold?.length ? `${cold.join(', ')}ms` : 'none over 50ms'}); longest rAF gap ${Math.max(0, ...items.map((i) => i.maxGap))}ms (limit LoAF ${LOAF_MAX_MS}ms)`,
		failures: items
			.filter((i) => i.status === 'FAIL')
			.map(
				(i) =>
					`${i.target}: long frames ${i.loaf.map((e) => `${e.ms}ms at ${typeof e.at === 'number' ? `${e.at}ms` : e.at}`).join(', ')}${i.window ? ` (settle ran ${i.window[0]}-${i.window[1]}ms)` : ''}`,
			),
		items,
	};
}

async function check6(ctx, base) {
	const items = [];
	// settle under reduced motion: tiles must never leave identity
	{
		const page = await openPage(ctx, base, { autostart: [SETTLE_TILES] });
		const run = await call(page, 'stop');
		const moved = changedFrames(run, 0, Number.POSITIVE_INFINITY);
		const off = run.frames.findIndex((f) =>
			f.v.some((v) => Math.abs(v[0]) + Math.abs(v[1]) > MOVING_EPS || Math.abs(v[5] - 1) > 1e-4),
		);
		items.push({
			target: 'bench settle',
			status: off >= 0 || moved.length ? 'FAIL' : 'PASS',
			detail:
				off >= 0
					? `transform off identity at frame ${off} (t=${r2(run.frames[off].t - run.frames[0].t)}ms)`
					: `${run.frames.length} frames, transform stayed identity`,
		});
		await page.close();
	}
	const page = await openPage(ctx, base);
	for (const t of await targets(page)) {
		const glide = t.kind === 'glide';
		const g = await call(page, 'geo', t.spec);
		if (!g || (glide && g.items.length < 2)) continue;
		await sleep(150);
		const mouse = pointer(page);
		const b = t.kind === 'tilt' ? g.box : g.el;
		const out = outside(g.host, g);
		const inside = glide
			? [cx(g.items[0]), cy(g.items[0])]
			: t.kind === 'deck'
				? [g.el.l + 2, cy(g.el)]
				: [b.l + b.w * 0.2, b.t + b.h * 0.2];
		const far = glide
			? [cx(g.items.at(-1)), cy(g.items.at(-1))]
			: t.kind === 'deck'
				? [footprint(g).r - 4, cy(g.el)]
				: [b.l + b.w * 0.8, b.t + b.h * 0.8];
		await mouse.to(...out);
		await sleep(REST_MS);
		const run = await sampled(
			page,
			[t.spec],
			async (mark) => {
				await mark('enter');
				await mouse.to(...inside);
				await sleep(500);
				await mark('move');
				await mouse.to(...far);
				await sleep(500);
				await mark('leave');
				await mouse.to(...out);
			},
			{ glide, after: 600 },
		);
		const ends = [...run.marks.map((m) => m.t), Number.POSITIVE_INFINITY];
		const per = run.marks.map((m, i) => ({
			name: m.name,
			frames: changedFrames(run, m.t, ends[i + 1], glide),
		}));
		const must = t.kind === 'deck' ? await call(page, 'state', t.spec) : null;
		const bad = per
			.filter((p) => p.frames.length > REDUCED_MAX_CHANGED_FRAMES)
			.map(
				(p) =>
					`'${p.name}' moved over ${p.frames.length} frames (first ${p.frames[0]}, last ${p.frames.at(-1)}, t=${r2(run.frames[p.frames[0]].t - run.frames[0].t)}ms)`,
			);
		let note = '';
		if (t.kind === 'deck') {
			const sawChange = per[0].frames.length >= 1;
			const closed = !must.open;
			if (!sawChange) bad.push('deck did not change pose on open');
			if (!closed) bad.push('deck still data-open after leave');
			note = `data-open state change ${sawChange ? 'seen' : 'MISSING'}`;
		}
		items.push({
			target: t.name,
			status: bad.length ? 'FAIL' : 'PASS',
			detail: `${per.map((p) => `${p.name} ${p.frames.length}f`).join(', ')}${note ? `; ${note}` : ''}`,
			bad,
		});
	}
	await page.close();
	return {
		status: verdict(items),
		summary: `${items.length} objects, most frames moved per event ${Math.max(0, ...items.map((i) => Number((i.detail.match(/(\d+)f/g) ?? ['0f']).map((s) => Number.parseInt(s, 10)).sort((a, b) => b - a)[0])))} (limit ${REDUCED_MAX_CHANGED_FRAMES})`,
		failures: items.flatMap((i) =>
			(i.bad ?? [])
				.concat(i.status === 'FAIL' && !i.bad ? [i.detail] : [])
				.map((b) => `${i.target}: ${b}`),
		),
		items,
	};
}

async function check7(ctx, base) {
	const page = await ctx.newPage();
	const specs = [
		{ name: 'deck', sel: '[data-deck]', i: 0, sub: 'cards' },
		{ name: 'deck2', sel: '[data-deck]', i: 1, sub: 'cards' },
		{ name: 'tilt', sel: '[data-tilt]', i: 0 },
		...[0, 1, 2, 3, 4].map((i) => ({
			name: `magnetic${i}`,
			sel: '[data-magnetic]',
			i,
			sub: 'child',
		})),
	];
	await page.addInitScript((s) => {
		window.__m.start(s);
		document.addEventListener('DOMContentLoaded', () => window.__m.attrs());
	}, specs);
	await page.goto(base, { waitUntil: 'load' });
	await sleep(1500);
	await page.evaluate(() => window.__m.scrollTour());
	await sleep(600);
	const run = await call(page, 'stop');
	const attrs = await page.evaluate(() => window.__m.attrCount());
	await page.close();
	const base0 = run.frames[0].v;
	const items = run.labels.map((label, k) => {
		let drift = 0;
		let at = -1;
		run.frames.forEach((f, i) => {
			const d = Math.max(
				Math.hypot(f.v[k][0] - base0[k][0], f.v[k][1] - base0[k][1]),
				Math.abs(f.v[k][2] - base0[k][2]),
				Math.abs(f.v[k][3] - base0[k][3]),
				Math.abs(f.v[k][4] - base0[k][4]),
			);
			if (d > drift) [drift, at] = [d, i];
		});
		return {
			target: label,
			status: drift > TOUCH_MAX_DRIFT ? 'FAIL' : 'PASS',
			drift: r2(drift),
			frame: at,
		};
	});
	if (attrs.n)
		items.push({ target: 'state attributes', status: 'FAIL', drift: attrs.n, names: attrs.names });
	return {
		status: verdict(items),
		summary: `${run.frames.length} frames over load + scroll, max drift ${Math.max(0, ...items.map((i) => i.drift))} (limit ${TOUCH_MAX_DRIFT}), ${attrs.n} data-open/active/tilt-active changes`,
		failures: items
			.filter((i) => i.status === 'FAIL')
			.map((i) =>
				i.names
					? `${i.drift} state attribute changes: ${i.names.join('; ')}`
					: `${i.target} moved ${i.drift} by frame ${i.frame} (t=${r2(run.frames[i.frame].t - run.frames[0].t)}ms)`,
			),
		items,
	};
}

// ---- scripted tour (video) -------------------------------------------------------------------------
async function tour(ctx, base, vp) {
	const page = await ctx.newPage();
	await page.addInitScript(() => {
		addEventListener('DOMContentLoaded', () => {
			const dot = document.createElement('div');
			dot.style.cssText =
				'position:fixed;z-index:99999;width:14px;height:14px;margin:-7px;border-radius:50%;background:#f00;opacity:.7;pointer-events:none;left:-50px;top:0';
			document.body.append(dot);
			addEventListener(
				'pointermove',
				(e) => {
					dot.style.left = `${e.clientX}px`;
					dot.style.top = `${e.clientY}px`;
				},
				true,
			);
		});
	});
	await page.goto(base, { waitUntil: 'load' });
	await page.waitForSelector('[data-glide-ready]');
	await sleep(1200);
	if (vp.hasTouch) {
		await page.evaluate(() => window.__m.scrollTour());
		await sleep(500);
		return page;
	}
	const mouse = pointer(page);
	for (const t of await targets(page)) {
		const g = await call(page, 'geo', t.spec);
		if (!g) continue;
		await sleep(250);
		const out = outside(g.host, g);
		await mouse.to(...out, 300);
		if (t.kind === 'deck') {
			const fp = footprint(g);
			const y = cy(g.el);
			await mouse.to(g.el.l + 2, y, 400);
			await mouse.to(fp.l + fp.w * 0.55, y, 1100);
			await mouse.to(fp.l + 6, y, 800);
			await mouse.to(fp.r - 4, y, 1400);
		} else if (t.kind === 'glide') {
			for (const it of g.items.slice(0, 6)) await mouse.to(cx(it), cy(it), 350);
		} else {
			const b = t.kind === 'tilt' ? g.box : g.el;
			for (const [u, v] of [
				[0.15, 0.15],
				[0.85, 0.2],
				[0.8, 0.85],
				[0.2, 0.8],
				[0.5, 0.5],
			])
				await mouse.to(b.l + b.w * u, b.t + b.h * v, 450);
		}
		await mouse.to(...out, 400);
		await sleep(400);
	}
	return page;
}

// ---- run ----------------------------------------------------------------------------------------
const CHECKS = [
	['1 interruption continuity', check1, (vp) => !vp.hasTouch],
	['2 zone stability', check2, (vp) => !vp.hasTouch],
	['3 layout reads per move', check3, (vp) => !vp.hasTouch],
	['4 animated properties', check4, () => true],
	['5 frame budget', check5, () => true],
	['6 reduced motion', check6, (vp) => !vp.hasTouch],
	['7 touch inertness', check7, (vp) => vp.hasTouch],
];

async function runEngine(engine, base) {
	const results = [];
	const browser = await ENGINES[engine].launch();
	try {
		for (const vp of VIEWPORTS) {
			const { name, width, height, ...emu } = vp;
			const viewport = { width, height };
			const ctx = await browser.newContext({ viewport, ...emu });
			const reduced = await browser.newContext({ viewport, reducedMotion: 'reduce', ...emu });
			for (const c of [ctx, reduced]) await c.addInitScript(instrument);
			for (const [title, fn, applies] of CHECKS) {
				if (process.env.MOTION_ONLY && !title.startsWith(process.env.MOTION_ONLY)) continue;
				const row = { engine, viewport: `${name} ${width}x${height}`, check: title };
				if (!applies(vp)) {
					results.push({
						...row,
						status: 'SKIP',
						summary: vp.hasTouch ? 'needs a hover pointer' : 'touch viewport only',
						items: [],
					});
					continue;
				}
				fpsSeen.length = 0;
				let out;
				// the dev server reloads the page when someone saves a source file: retry once
				for (let attempt = 0; attempt < 3; attempt++) {
					const before = reloads;
					try {
						out = await fn(
							title.startsWith('6') ? reduced : ctx,
							base,
							title.startsWith('4') ? vp.hasTouch : engine,
						);
						if (reloads === before) break;
						out.note = 'page reloaded by the dev server mid-run, rerun';
					} catch (error) {
						out = {
							status: 'FAIL',
							summary: `check crashed: ${error.message.split('\n')[0]}`,
							failures: [error.stack],
							items: [],
						};
						if (
							reloads === before &&
							!/context was destroyed|navigation|is not a function/.test(error.message)
						)
							break;
					}
				}
				const fps = fpsSeen.length ? r2(median(fpsSeen)) : null;
				const last = { ...row, ...out, fps };
				results.push(last);
				console.log(
					`${engine} ${name.padEnd(7)} ${last.status.padEnd(4)} ${title.padEnd(26)} ${last.summary}${fps ? ` [${fps}fps]` : ''}`,
				);
			}
			await ctx.close();
			await reduced.close();
			// the video context is separate so recording never slows the measured runs
			const dir = `${OUT}/video-tmp`;
			const rec = await browser.newContext({
				viewport,
				recordVideo: { dir, size: viewport },
				...emu,
			});
			await rec.addInitScript(instrument);
			const page = await tour(rec, base, vp);
			const video = page.video();
			await rec.close();
			const file = `${OUT}/${engine}-${name}.webm`;
			await video.saveAs(file);
			await video.delete();
			results.push({
				engine,
				viewport: `${name} ${width}x${height}`,
				check: 'video',
				status: 'INFO',
				summary: file,
				items: [],
			});
			console.log(`${engine} ${name.padEnd(7)} video ${file}`);
		}
	} finally {
		await browser.close();
	}
	return results;
}

const MIME = {
	'.html': 'text/html',
	'.js': 'text/javascript',
	'.css': 'text/css',
	'.svg': 'image/svg+xml',
	'.json': 'application/json',
	'.xml': 'application/xml',
	'.jpg': 'image/jpeg',
	'.png': 'image/png',
	'.webp': 'image/webp',
	'.avif': 'image/avif',
	'.woff2': 'font/woff2',
	'.glb': 'model/gltf-binary',
};

/** Builds once per run (the `all` parent builds; its children inherit MOTION_BUILT), serves dist/. */
async function serveDist() {
	if (!process.env.MOTION_BUILT && !process.env.MOTION_NOBUILD) {
		const b = spawnSync(process.execPath, ['node_modules/astro/astro.js', 'build'], {
			stdio: 'inherit',
		});
		if (b.status !== 0) process.exit(1);
	}
	process.env.MOTION_BUILT = '1';
	const server = createServer(async (req, res) => {
		const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(
			/^(\.\.[/\\])+/,
			'',
		);
		let file = join('dist', path);
		try {
			if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
			const body = await readFile(file);
			res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' });
			res.end(body);
		} catch {
			res.writeHead(404).end();
		}
	});
	await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
	return { base: `http://127.0.0.1:${server.address().port}`, stop: () => server.close() };
}

async function resolveBase() {
	if (process.env.DEV_URL) return { base: process.env.DEV_URL.replace(/\/$/, ''), stop: () => {} };
	if (!DEV) return serveDist();
	try {
		if ((await fetch(DEFAULT_URL, { signal: AbortSignal.timeout(2000) })).ok)
			return { base: DEFAULT_URL, stop: () => {} };
	} catch {}
	return serve();
}

if (arg === 'all') {
	let failed = false;
	let stopBuild = () => {};
	if (!DEV && !process.env.DEV_URL) stopBuild = (await serveDist()).stop;
	stopBuild();
	for (const engine of Object.keys(ENGINES)) {
		if (!installed(engine)) {
			console.log(`${engine}: not installed, skipped (not installing it)`);
			continue;
		}
		failed ||=
			spawnSync(
				process.execPath,
				[process.argv[1], engine, ...process.argv.slice(2).filter((a) => a === '--dev')],
				{ stdio: 'inherit' },
			).status !== 0;
	}
	process.exit(failed ? 1 : 0);
}
if (!installed(arg)) {
	console.log(`${arg}: not installed, skipped (not installing it)`);
	process.exit(0);
}

await mkdir(OUT, { recursive: true });
const { base, stop } = await resolveBase();
setTimeout(() => {
	console.error('motion check did not finish within 15 minutes');
	stop();
	process.exit(1);
}, 900_000).unref();
let results = [];
try {
	results = await runEngine(arg, `${base}/`);
} finally {
	stop();
}
await writeFile(
	`${OUT}/report-${arg}.json`,
	JSON.stringify({ engine: arg, base, results }, null, 2),
);
const fails = results.filter((r) => r.status === 'FAIL');
for (const r of fails) {
	console.log(`\nFAIL ${r.engine} ${r.viewport} ${r.check}`);
	for (const f of (r.failures ?? []).slice(0, 12)) console.log(`  - ${f.split('\n')[0]}`);
	if ((r.failures ?? []).length > 12)
		console.log(`  ... ${r.failures.length - 12} more in the JSON report`);
}
console.log(
	`\n${arg}: ${results.filter((r) => r.status === 'PASS').length} pass, ${fails.length} fail; report ${OUT}/report-${arg}.json`,
);
process.exit(fails.length ? 1 : 0);
