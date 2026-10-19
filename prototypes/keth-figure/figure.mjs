// Keth bench: the static drawing. Builds SVG strings with the anatomy kit; no runtime deps.
import * as k from './kit/iso-kit.mjs';
import { BRANCHES, cardWidth, L } from './model.mjs';

export const WIDTH = 780;
export const HEIGHT = 350;

const PLATE = { x: -256, y: -114, w: 528, d: 228, r: 10 };
const PLATE_H = 12;
const FOOT_H = 5;
const BEAM = { z0: 36, z1: 56, half: 5, y: 110 };
const RAIL_Y = 104;

const points = [
	...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0),
	[252, -2, 70],
	[-6, -112, 58],
	[-6, 112, 58],
];
export const P = k.fitProjection(points, WIDTH, HEIGHT, { pad: 10, azimuth: 60, elevation: 34 });

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const Ln = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const slab = (plan, z, h, tone = 'mid', steps = 3, bevel = 0, extra = {}) =>
	S(k.slabOf(plan, z, h, P, steps, bevel), { tone, ...extra });
const cyl = (x, y, r, z, h, tone = 'mid', steps = 24, bevel = 0, extra = {}) =>
	S(k.cylinder(x, y, r, z, h, P, steps, bevel), { tone, ...extra });
const byDepth = (pts) =>
	[...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const g = (attrs, body) =>
	`<g ${Object.entries(attrs)
		.map(([key, value]) => `${key}="${value}"`)
		.join(' ')}>${body}</g>`;
const dotAt = (x, y, z) => at([x, y, z]);

function plate() {
	const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
	for (const [x, y] of byDepth(k.corners(PLATE, 24))) {
		out.push(cyl(x, y, 8, -PLATE_H - FOOT_H, 1.6, 'lo', 28));
		out.push(cyl(x, y, 6.2, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, 'mid', 28));
	}
	out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: 'mid' }));
	out.push(Ln(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: 'lo' }));
	const screws = k.corners(PLATE, 11);
	out.push(Ln(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(''), { tone: 'lo' }));
	out.push(
		D(
			screws.map(([x, y]) => dotAt(x, y, 0)),
			{ size: 0.5 },
		),
	);
	// label plate near the right front corner
	const lp = { x: 226, y: 80, w: 32, d: 20, r: 1.6 };
	out.push(slab(lp, 0, 0.8, 'lo', 4));
	out.push(Ln(k.planOutline(k.insetPlan(lp, 1.8), 0.8, P, 4), { tone: 'faint' }));
	for (const [i, len] of [
		[0, 18],
		[1, 13],
		[2, 16],
	]) {
		const yy = lp.y + 5.6 + i * 3.8;
		out.push(
			Ln(k.lineOnTop([lp.x + 6, yy], [lp.x + 6 + len, yy], 0.8, P), { tone: 'lo', free: true }),
		);
	}
	out.push(
		D([dotAt(lp.x + 3.2, lp.y + 3.2, 0.8), dotAt(lp.x + lp.w - 3.2, lp.y + lp.d - 3.2, 0.8)], {
			size: 0.4,
		}),
	);
	return out.join('');
}

function rail(y, name) {
	const out = [];
	const plan = { x: -250, y: y - 2.5, w: 514, d: 5, r: 1 };
	out.push(slab(plan, 0, 3, 'mid', 3, 0.4));
	out.push(
		D(
			[-232, -120, 0, 120, 232].map((x) => dotAt(x, y, 3)),
			{ size: 0.4 },
		),
	);
	for (const x of [-256, 262]) {
		out.push(slab({ x: x - 2.5, y: y - 6, w: 7, d: 12, r: 1.2 }, 0, 8, 'mid', 3, 0.5));
		out.push(D([dotAt(x + 1, y - 2.5, 8), dotAt(x + 1, y + 2.5, 8)], { size: 0.4 }));
	}
	return g({ 'data-part': name }, out.join(''));
}

function carriageAt(y) {
	const out = [];
	out.push(slab({ x: -10, y: y - 6.5, w: 20, d: 13, r: 1.6 }, 3, 6, 'mid', 4, 0.6));
	out.push(
		D([dotAt(-6, y - 3, 9), dotAt(6, y - 3, 9), dotAt(-6, y + 3, 9), dotAt(6, y + 3, 9)], {
			size: 0.4,
			tone: 'lo',
		}),
	);
	out.push(slab({ x: -6, y: y - 4, w: 12, d: 8, r: 1.4 }, 9, BEAM.z0 - 9, 'mid', 3, 0.5));
	out.push(Ln(k.sideSeam({ x: -6, y: y - 4, w: 12, d: 8, r: 1.4 }, 18, P, 3), { tone: 'faint' }));
	return out.join('');
}

// the block strip's lane: channel, ruler, cards
const LANE = { x: -232, y: L.CARD_Y - 21, w: 402, d: 42, r: 3 };
function lane() {
	const far = [];
	far.push(slab(LANE, 0, 1.5, 'lo', 4, 0.3));
	far.push(slab({ x: LANE.x, y: LANE.y, w: LANE.w, d: 2.4, r: 0.6 }, 1.5, 3.4, 'mid', 2, 0.3));
	const near = [];
	near.push(
		slab(
			{ x: LANE.x, y: LANE.y + LANE.d - 2.4, w: LANE.w, d: 2.4, r: 0.6 },
			1.5,
			3.4,
			'mid',
			2,
			0.3,
		),
	);
	near.push(
		D(
			[-224, -100, 0, 100, 160].map((x) => dotAt(x, LANE.y + LANE.d - 1.2, 4.9)),
			{ size: 0.38 },
		),
	);
	// ruler strip: one minor tick per transaction, a major per body chunk
	const ruler = { x: L.X0 + L.INIT_W - 5, y: 90, w: 5 * L.BODY_W + 10, d: 7, r: 1 };
	near.push(slab(ruler, 0, 0.6, 'lo', 3));
	const ticks = k.topTicks(
		L.X0 + L.INIT_W,
		L.X0 + L.INIT_W + 5 * L.BODY_W,
		5,
		10,
		ruler.y + 0.8,
		0.6,
		[2, 3.6],
		P,
	);
	near.push(Ln(ticks.minor, { tone: 'lo' }), Ln(ticks.major, { tone: 'mid' }));
	near.push(
		D([dotAt(ruler.x + 2, ruler.y + 3.5, 0.6), dotAt(ruler.x + ruler.w - 2, ruler.y + 3.5, 0.6)], {
			size: 0.42,
		}),
	);
	return { far: far.join(''), near: near.join('') };
}

// the ring a stamp leaves on a card
const impression = (cx) =>
	g(
		{ 'data-imp': '' },
		Ln(
			k.ring(cx, L.CARD_Y + 2, 5.2, L.CARD_TOP, P, 24) +
				k.ring(cx, L.CARD_Y + 2, 2.6, L.CARD_TOP, P, 16),
			{ tone: 'hi' },
		),
	);
function card(kind, x, withImpression = true) {
	const w = kind === 'init' ? L.INIT_W : kind === 'tear' ? L.TEAR_W : L.BODY_W;
	const y = L.CARD_Y - L.CARD_D / 2;
	const plan = { x, y, w, d: L.CARD_D, r: 2 };
	const out = [slab(plan, 1.5, 5, 'hi', 4, 0.5)];
	out.push(Ln(k.planOutline(k.insetPlan(plan, 1.6), L.CARD_TOP, P, 4), { tone: 'faint' }));
	if (kind === 'body') {
		// ten cells: one per transaction
		let cells = '';
		for (let c = 0; c < 10; c++) {
			const cell = { x: x + 3.8 + c * 4.34, y: y + 5.2, w: 3.1, d: 14, r: 0.7 };
			cells += k.planOutline(cell, L.CARD_TOP, P, 2);
		}
		out.push(Ln(cells, { tone: 'lo' }));
		out.push(
			D([dotAt(x + 4, y + 26, L.CARD_TOP), dotAt(x + w - 4, y + 26, L.CARD_TOP)], { size: 0.45 }),
		);
	} else if (kind === 'init') {
		for (const [i, len] of [
			[0, 20],
			[1, 14],
			[2, 17],
		]) {
			const yy = y + 6.5 + i * 3.6;
			out.push(
				Ln(k.lineOnTop([x + 5, yy], [x + 5 + len, yy], L.CARD_TOP, P), { tone: 'lo', free: true }),
			);
		}
		out.push(
			D([dotAt(x + 4.4, y + 26, L.CARD_TOP), dotAt(x + w - 4.4, y + 26, L.CARD_TOP)], {
				size: 0.45,
			}),
		);
	} else {
		let slots = '';
		for (let c = 0; c < 3; c++)
			slots += k.planOutline(
				{ x: x + 5 + c * 8.6, y: y + 6, w: 5.6, d: 10, r: 1.4 },
				L.CARD_TOP,
				P,
				3,
			);
		out.push(Ln(slots, { tone: 'lo' }));
		out.push(
			D([dotAt(x + 4.4, y + 26, L.CARD_TOP), dotAt(x + w - 4.4, y + 26, L.CARD_TOP)], {
				size: 0.45,
			}),
		);
	}
	if (withImpression) out.push(impression(x + w / 2));
	return out.join('');
}

// the 16-slot branch tray at the back
const TRAY = { x: -3, y: -98, w: 196, d: 56, r: 2 };
const slotCardBody = () => {
	const plan = { x: 0, y: TRAY.y + 4, w: 9.2, d: 46, r: 1 };
	let rings = '';
	for (let b = 0; b < 4; b++) rings += k.ring(4.6, TRAY.y + 9 + b * 4.6, 1.1, L.CARD_TOP, P, 10);
	return (
		slab(plan, 1.5, 5, 'hi', 3, 0.4) +
		Ln(k.planOutline(k.insetPlan(plan, 1.4), L.CARD_TOP, P, 3), { tone: 'faint' }) +
		Ln(rings, { tone: 'lo' })
	);
};
const dividerBody = () =>
	slab({ x: -0.8, y: TRAY.y + 2.6, w: 1.6, d: 51, r: 0.3 }, 1.5, 6, 'mid', 2, 0.2);
function tray() {
	const out = [];
	out.push(slab(TRAY, 0, 1.5, 'lo', 3, 0.3));
	out.push(slab({ x: TRAY.x, y: TRAY.y, w: TRAY.w, d: 2.6, r: 0.6 }, 1.5, 6.5, 'mid', 2, 0.4));
	out.push(
		Ln(k.lineOnTop([TRAY.x + 4, TRAY.y + 1.3], [TRAY.x + TRAY.w - 4, TRAY.y + 1.3], 8, P), {
			tone: 'faint',
		}),
	);
	for (let j = 0; j <= BRANCHES; j++) {
		const x = L.SLOT * j;
		if (j > 0) {
			const cardX = L.SLOT * (j - 1) + 1.4;
			const parts = [useOf('k-slotcard', [cardX, 0, 0])];
			// the branch index as a nibble: four holes, filled where the bit is set
			const idx = j - 1;
			const cx = cardX + 4.6;
			const bits = [];
			for (let b = 0; b < 4; b++) {
				if ((idx >> (3 - b)) & 1) bits.push(dotAt(cx, TRAY.y + 9 + b * 4.6, L.CARD_TOP));
			}
			if (bits.length) parts.push(D(bits, { size: 0.55, tone: 'hi' }));
			parts.push(
				g(
					{ 'data-imp': '' },
					Ln(
						k.ring(cx, TRAY.y + 38, 3.2, L.CARD_TOP, P, 18) +
							k.ring(cx, TRAY.y + 38, 1.4, L.CARD_TOP, P, 12),
						{ tone: 'hi' },
					),
				),
			);
			out.push(g({ 'data-slot': idx }, parts.join('')));
		}
		out.push(useOf('k-div', [x, 0, 0]));
	}
	out.push(
		slab(
			{ x: TRAY.x, y: TRAY.y + TRAY.d - 2.6, w: TRAY.w, d: 2.6, r: 0.6 },
			1.5,
			2.4,
			'mid',
			2,
			0.3,
		),
	);
	// nameplate in front of the tray, the one engraved label
	const np = { x: 70, y: -39, w: 52, d: 9, r: 1.4 };
	out.push(slab(np, 0, 0.8, 'lo', 4));
	out.push(Ln(k.planOutline(k.insetPlan(np, 1.2), 0.8, P, 4), { tone: 'faint' }));
	out.push(
		k.faceTextSvg(k.topMatrix([np.x + np.w / 2, np.y + 6.2], 0.8, P, 'x'), 'mpt_diff', {
			size: 4.4,
			tone: 'mid',
			anchor: 'middle',
		}),
	);
	out.push(
		D([dotAt(np.x + 2.6, np.y + 4.5, 0.8), dotAt(np.x + np.w - 2.6, np.y + 4.5, 0.8)], {
			size: 0.4,
		}),
	);
	return out.join('');
}

// the seal trough: pockets the seals land in, key slots between them
const TROUGH = { x: -152, y: L.TROUGH_Y - 12, w: 334, d: 24, r: 3 };
const FLOOR = 2;
function trough() {
	const out = [];
	out.push(slab(TROUGH, 0, FLOOR, 'lo', 4, 0.3));
	out.push(
		slab({ x: TROUGH.x, y: TROUGH.y, w: TROUGH.w, d: 2, r: 0.5 }, FLOOR, 2.2, 'mid', 2, 0.3),
	);
	const near = slab(
		{ x: TROUGH.x, y: TROUGH.y + TROUGH.d - 2, w: TROUGH.w, d: 2, r: 0.5 },
		FLOOR,
		2.2,
		'mid',
		2,
		0.3,
	);
	let pockets = '';
	let slots = '';
	for (let p = 0; p < L.POCKETS; p++) {
		const x = L.XR - L.POCKET * p;
		pockets += k.ring(x, L.TROUGH_Y, 5.4, FLOOR, P, 24);
		if (p < L.POCKETS - 1)
			slots += k.planOutline(
				{ x: x - L.POCKET / 2 - 1.8, y: L.TROUGH_Y - 2.6, w: 3.6, d: 5.2, r: 0.8 },
				FLOOR,
				P,
				2,
			);
	}
	out.push(Ln(pockets, { tone: 'lo' }), Ln(slots, { tone: 'faint' }));
	return { body: out.join(''), near };
}

const useOf = (id, by) =>
	`<use href="#${id}"${by ? ` transform="${k.translateAlong(P, by)}"` : ''}/>`;
// repeated parts are drawn once in <defs> and placed with <use>: the page stays small
const tokenBody = () =>
	S(k.cylinder(0, 0, 4.6, 0, 2.2, P, 24, 0.5), { tone: 'hi' }) +
	Ln(k.ring(0, 0, 3.4, 2.2, P, 20), { tone: 'lo' });
function token(kIndex) {
	// each seal is notched differently: three dots on an inner ring
	const dots = [];
	for (let i = 0; i < 3; i++) {
		const a = ((kIndex * 37 + i * 119) % 360) * (Math.PI / 180);
		dots.push(dotAt(Math.cos(a) * 2.1, Math.sin(a) * 2.1, 2.2));
	}
	return g(
		{ 'data-token': kIndex },
		g({ 'data-move': '' }, useOf('k-token') + D(dots, { size: 0.5, tone: 'hi' })),
	);
}

const keyBody = () =>
	slab({ x: -4.2, y: L.TROUGH_Y - 1.8, w: 8.4, d: 3.6, r: 0.8 }, FLOOR + 2.2, 1.2, 'mid', 2, 0.2);
function keyBridge(p) {
	const x = L.XR - L.POCKET * p - L.POCKET / 2;
	return g({ 'data-key': p }, useOf('k-key', [x, 0, 0]));
}

// the comb that checks every joint at once, hung from a column at the right end of the trough
const COL = { x: 188, y: L.TROUGH_Y - 6 };
const COMB = { bottom: 11.4, top: 16.4 };
function combFrame() {
	const out = [];
	out.push(slab({ x: COL.x - 4, y: COL.y - 4, w: 20, d: 20, r: 2 }, 0, 6, 'mid', 3, 0.5));
	out.push(
		D(
			[
				dotAt(COL.x - 1, COL.y - 1, 6),
				dotAt(COL.x + 13, COL.y - 1, 6),
				dotAt(COL.x - 1, COL.y + 13, 6),
				dotAt(COL.x + 13, COL.y + 13, 6),
			],
			{ size: 0.4 },
		),
	);
	out.push(slab({ x: COL.x, y: COL.y, w: 12, d: 12, r: 2 }, 6, 34, 'mid', 3, 0.5));
	out.push(Ln(k.sideSeam({ x: COL.x, y: COL.y, w: 12, d: 12, r: 2 }, 20, P, 3), { tone: 'faint' }));
	out.push(
		slab({ x: COL.x - 0.8, y: COL.y - 0.8, w: 13.6, d: 13.6, r: 2.4 }, 40, 3, 'mid', 3, 0.4),
	);
	return out.join('');
}
function comb() {
	const x0 = L.XR - L.POCKET * (L.POCKETS - 1) - 12;
	const x1 = COL.x;
	const bar = { x: x0, y: L.TROUGH_Y - 5, w: x1 - x0, d: 10, r: 1.4 };
	const out = [];
	for (let p = 0; p < L.POCKETS - 1; p++) {
		const x = L.XR - L.POCKET * p - L.POCKET / 2;
		out.push(g({ 'data-pin': p }, useOf('k-pin', [x, 0, 0])));
	}
	out.push(slab(bar, COMB.bottom, COMB.top - COMB.bottom, 'mid', 3, 0.6));
	out.push(Ln(k.planOutline(k.insetPlan(bar, 2.2), COMB.top, P, 3), { tone: 'faint' }));
	out.push(
		D(
			[-120, -60, 0, 60, 120].map((x) => dotAt(x, L.TROUGH_Y, COMB.top)),
			{ size: 0.4, tone: 'lo' },
		),
	);
	// carrier plate on the column's far face
	out.push(
		slab({ x: COL.x - 3, y: COL.y + 1, w: 3, d: 10, r: 0.6 }, COMB.bottom - 3, 12, 'mid', 2, 0.3),
	);
	return out.join('');
}

// the lead press at the right end: anvil, post, arm, ram, proof seal and its lamp
const PRESS = { px: 250, pz: L.TROUGH_Y };
function pressStatic() {
	const out = [];
	out.push(slab({ x: 236, y: -16, w: 32, d: 28, r: 3 }, 0, 8, 'mid', 4, 0.8));
	out.push(Ln(k.planOutline({ x: 239, y: -13, w: 26, d: 22, r: 2 }, 8, P, 3), { tone: 'faint' }));
	out.push(
		D([dotAt(241, -10, 8), dotAt(263, -10, 8), dotAt(241, 5, 8), dotAt(263, 5, 8)], { size: 0.45 }),
	);
	// anvil pad the proof seal sits on
	out.push(
		slab(
			{ x: L.PROOF_X - 13, y: L.TROUGH_Y - 10, w: 26, d: 20, r: 2.4 },
			0,
			FLOOR + 0.4,
			'mid',
			4,
			0.4,
		),
	);
	out.push(Ln(k.ring(L.PROOF_X, L.TROUGH_Y, 8.6, FLOOR + 0.4, P, 28), { tone: 'lo' }));
	return out.join('');
}
function pressPost() {
	const out = [];
	out.push(slab({ x: PRESS.px - 6, y: L.TROUGH_Y - 6, w: 12, d: 12, r: 2 }, 8, 42, 'mid', 3, 0.5));
	out.push(
		Ln(k.sideSeam({ x: PRESS.px - 6, y: L.TROUGH_Y - 6, w: 12, d: 12, r: 2 }, 24, P, 3), {
			tone: 'faint',
		}),
	);
	out.push(
		slab({ x: PRESS.px - 7, y: L.TROUGH_Y - 7, w: 14, d: 14, r: 2.4 }, 50, 4, 'mid', 3, 0.4),
	);
	out.push(
		slab(
			{ x: L.PROOF_X - 11, y: L.TROUGH_Y - 5, w: PRESS.px - L.PROOF_X + 6, d: 10, r: 1.6 },
			50,
			4,
			'mid',
			3,
			0.5,
		),
	);
	out.push(
		D([dotAt(L.PROOF_X - 4, L.TROUGH_Y, 54), dotAt(PRESS.px - 12, L.TROUGH_Y, 54)], { size: 0.45 }),
	);
	return out.join('');
}
const RAM = { bottom: 5.3 };
function ram() {
	const x = L.PROOF_X;
	const y = L.TROUGH_Y;
	const out = [];
	out.push(cyl(x, y, 7.8, RAM.bottom, 3, 'mid', 28, 0.4));
	out.push(cyl(x, y, 2.4, RAM.bottom + 3, 36, 'mid', 16, 0));
	return out.join('');
}
function ramKnob() {
	const x = L.PROOF_X;
	const y = L.TROUGH_Y;
	const out = [];
	out.push(cyl(x, y, 2.2, 54, 8, 'mid', 16, 0));
	out.push(cyl(x, y, 5, 62, 4.4, 'mid', 28, 0.4));
	out.push(Ln(k.knurl(x, y, 5, 62, 66.4, 32, P), { tone: 'lo' }));
	out.push(Ln(k.lineOnTop([x - 2.4, y], [x + 3.6, y], 66.4, P), { tone: 'hi' }));
	return out.join('');
}
function proofSeal() {
	const x = L.PROOF_X;
	const y = L.TROUGH_Y;
	const z = FLOOR + 0.4;
	const out = [S(k.cylinder(x, y, 7.2, z, 2.8, P, 32, 0.5), { tone: 'hi' })];
	out.push(Ln(k.ring(x, y, 5.6, z + 2.8, P, 28), { tone: 'lo' }));
	out.push(Ln(k.ring(x, y, 2.8, z + 2.8, P, 18), { tone: 'mid' }));
	return out.join('');
}
function lamp() {
	const x = PRESS.px;
	const y = L.TROUGH_Y;
	return [cyl(x, y, 4.4, 54, 1.6, 'lo', 20, 0.3), cyl(x, y, 3, 55.6, 4.4, 'hi', 20, 0.3)].join('');
}

// the machine's controller, at the back left
function controlBox() {
	const plan = { x: -246, y: -92, w: 52, d: 40, r: 3 };
	const top = 16;
	const out = [slab(plan, 0, top, 'mid', 4, 0.8)];
	out.push(Ln(k.sideSeam(plan, 8, P, 4), { tone: 'faint' }));
	out.push(Ln(k.planOutline(k.insetPlan(plan, 3), top, P, 4), { tone: 'faint' }));
	let vents = '';
	for (let i = 0; i < 5; i++)
		vents += k.planOutline(
			{ x: plan.x + 7, y: plan.y + 8 + i * 4.6, w: 20, d: 2, r: 1 },
			top,
			P,
			3,
		);
	out.push(Ln(vents, { tone: 'lo' }));
	for (const [kx, ky] of [
		[plan.x + 38, plan.y + 11],
		[plan.x + 38, plan.y + 27],
	]) {
		out.push(cyl(kx, ky, 4.2, top, 3.6, 'mid', 24, 0.4));
		out.push(Ln(k.knurl(kx, ky, 4.2, top, top + 3.6, 24, P), { tone: 'lo' }));
		out.push(Ln(k.lineOnTop([kx - 1.6, ky], [kx + 3.4, ky], top + 3.6, P), { tone: 'hi' }));
	}
	out.push(cyl(plan.x + 12, plan.y + 31, 2.2, top, 1.8, 'lo', 16, 0.2));
	out.push(Ln(k.ring(plan.x + 12, plan.y + 31, 3.4, top, P, 16), { tone: 'lo' }));
	out.push(
		D([dotAt(plan.x + 5, plan.y + 5, top), dotAt(plan.x + plan.w - 5, plan.y + plan.d - 5, top)], {
			size: 0.45,
		}),
	);
	return out.join('');
}

// the cassette the block feeds from, at the left end of the card lane
function cassette() {
	const plan = { x: -254, y: L.CARD_Y - 20, w: 26, d: 40, r: 3 };
	const top = 14;
	const out = [slab(plan, 0, top, 'mid', 4, 0.8)];
	out.push(Ln(k.sideSeam(plan, 7, P, 4), { tone: 'faint' }));
	out.push(Ln(k.planOutline(k.insetPlan(plan, 3), top, P, 4), { tone: 'faint' }));
	for (const [i, len] of [
		[0, 14],
		[1, 9],
		[2, 12],
	]) {
		const xx = plan.x + 6 + i * 4.4;
		out.push(
			Ln(k.lineOnTop([xx, plan.y + 7], [xx, plan.y + 7 + len], top, P), { tone: 'lo', free: true }),
		);
	}
	out.push(
		D(
			[
				dotAt(plan.x + 4.4, plan.y + 4.4, top),
				dotAt(plan.x + 4.4, plan.y + plan.d - 4.4, top),
				dotAt(plan.x + plan.w - 4.4, plan.y + 4.4, top),
				dotAt(plan.x + plan.w - 4.4, plan.y + plan.d - 4.4, top),
			],
			{ size: 0.45 },
		),
	);
	return out.join('');
}

// gantry parts, built at beam x = 0
function beam() {
	const out = [];
	const plan = { x: -BEAM.half, y: -BEAM.y, w: BEAM.half * 2, d: BEAM.y * 2, r: 1 };
	out.push(slab(plan, BEAM.z0, BEAM.z1 - BEAM.z0, 'mid', 3, 0.8));
	out.push(Ln(k.lineOnTop([0, -BEAM.y], [0, BEAM.y], BEAM.z1, P), { tone: 'faint' }));
	out.push(Ln(k.sideSeam(plan, 46, P, 3), { tone: 'faint' }));
	out.push(
		D(
			[-100, -60, -20, 20, 60, 100].map((y) => dotAt(0, y, BEAM.z1)),
			{ size: 0.4, tone: 'lo' },
		),
	);
	// tool guide rails on the near face
	for (const yt of [L.CARD_Y, L.TRAY_Y]) {
		for (const dy of [-6, 6])
			out.push(
				slab({ x: BEAM.half, y: yt + dy - 0.8, w: 1.6, d: 1.6, r: 0.3 }, 30, 34, 'lo', 2, 0),
			);
	}
	return out.join('');
}
const STAMP = { bottom: L.CARD_TOP, x: L.STAMP_DX };
const ARM_Z = 36;
function stampAt(yt) {
	const z = STAMP.bottom;
	const out = [];
	out.push(cyl(STAMP.x, yt, 6, z, 3, 'mid', 24, 0.4));
	out.push(Ln(k.ring(STAMP.x, yt, 3.6, z + 3, P, 18), { tone: 'lo' }));
	out.push(cyl(STAMP.x, yt, 2, z + 3, ARM_Z - z - 3, 'mid', 14, 0));
	out.push(Ln(k.sideArc(STAMP.x, yt, 2, 24, P, 12), { tone: 'faint' }));
	out.push(slab({ x: BEAM.half + 1.6, y: yt - 6, w: 3, d: 12, r: 0.8 }, ARM_Z, 16, 'mid', 2, 0.4));
	out.push(
		slab(
			{ x: BEAM.half + 4.6, y: yt - 4, w: STAMP.x + 5 - BEAM.half - 4.6, d: 8, r: 1.2 },
			ARM_Z,
			4,
			'mid',
			3,
			0.4,
		),
	);
	out.push(
		D([dotAt(BEAM.half + 3, yt - 3.4, ARM_Z + 16), dotAt(BEAM.half + 3, yt + 3.4, ARM_Z + 16)], {
			size: 0.4,
		}),
	);
	return out.join('');
}
const BLADE_X = L.BLADE_DX;
const BLADE_YS = [L.CARD_Y - 12, L.CARD_Y + 12];
const BLADE_Y0 = L.CARD_Y - L.CARD_D / 2 - 1;
function blade() {
	const top = L.CARD_TOP + 20;
	const out = [];
	for (const yy of BLADE_YS) out.push(cyl(BLADE_X, yy, 1.2, top + 6, 12, 'hi', 12, 0));
	out.push(
		slab(
			{ x: BLADE_X - 1.6, y: BLADE_Y0, w: 3.2, d: L.CARD_D + 2, r: 0.4 },
			L.CARD_TOP,
			20,
			'hi',
			2,
			0.3,
		),
	);
	out.push(
		slab(
			{ x: BLADE_X - 2.4, y: BLADE_Y0 - 3, w: 4.8, d: L.CARD_D + 8, r: 0.8 },
			top,
			6,
			'mid',
			2,
			0.4,
		),
	);
	out.push(
		D(
			[
				dotAt(BLADE_X + 2.4, BLADE_Y0, top + 6),
				dotAt(BLADE_X + 2.4, BLADE_Y0 + L.CARD_D + 2, top + 6),
			],
			{ size: 0.42 },
		),
	);
	return out.join('');
}
// fixed guide blocks the blade's rods slide in
function bladeGuides() {
	return BLADE_YS.map((yy) =>
		slab({ x: BEAM.half, y: yy - 3.4, w: 6, d: 6.8, r: 1 }, 41, 23, 'mid', 3, 0.5),
	).join('');
}

export function buildFigure() {
	const lanes = lane();
	const tr = trough();
	const tokens = Array.from({ length: L.POCKETS }, (_, i) => token(i)).join('');
	const keys = Array.from({ length: L.POCKETS - 1 }, (_, p) => keyBridge(p)).join('');
	const cards = [
		g({ 'data-card': 0 }, card('init', L.X0)),
		...Array.from({ length: 5 }, (_, i) =>
			g(
				{ 'data-card': i + 1 },
				useOf('k-body', [L.X0 + L.INIT_W + L.BODY_W * i, 0, 0]) +
					impression(L.X0 + L.INIT_W + L.BODY_W * i + L.BODY_W / 2),
			),
		),
		g({ 'data-card': 't' }, card('tear', L.X0 + L.INIT_W + L.BODY_W)),
	].join('');
	const defs = `<defs>${g({ id: 'k-token' }, tokenBody())}${g({ id: 'k-key' }, keyBody())}${g({ id: 'k-pin' }, cyl(0, L.TROUGH_Y, 1.9, COMB.bottom - 6.2 + 0.4, 6.2, 'hi', 14, 0))}${g({ id: 'k-body' }, card('body', 0, false))}${g({ id: 'k-slotcard' }, slotCardBody())}${g({ id: 'k-div' }, dividerBody())}${g({ id: 'k-stamp' }, stampAt(0))}</defs>`;
	const body = [
		defs,
		plate(),
		rail(-RAIL_Y, 'rail-far'),
		g({ 'data-g': 'gantry' }, carriageAt(-RAIL_Y)),
		controlBox(),
		tray(),
		tr.body,
		lanes.far,
		cassette(),
		g({ 'data-fade': '' }, cards),
		lanes.near,
		tr.near,
		g({ 'data-fade': '' }, tokens),
		g({ 'data-fade': '' }, keys),
		combFrame(),
		g({ 'data-g': 'comb' }, comb()),
		pressStatic(),
		g({ 'data-fade': '', 'data-proof': '' }, proofSeal()),
		g({ 'data-g': 'ram' }, ram()),
		pressPost(),
		g({ 'data-g': 'ram' }, ramKnob()),
		g({ 'data-lamp': '' }, lamp()),
		rail(RAIL_Y, 'rail-near'),
		g({ 'data-g': 'gantry' }, carriageAt(RAIL_Y)),
		g({ 'data-g': 'gantry' }, beam()),
		g({ 'data-g': 'gantry' }, g({ 'data-g': 'blade' }, blade()) + bladeGuides()),
		g(
			{ 'data-g': 'gantry' },
			g({ 'data-g': 'stampF' }, useOf('k-stamp', [0, L.CARD_Y, 0])) +
				g({ 'data-g': 'stampB' }, useOf('k-stamp', [0, L.TRAY_Y, 0])),
		),
	];
	const svg = k.figureSvg({
		width: WIDTH,
		height: HEIGHT,
		label:
			'A bench machine. A strip of cells, the block, is cut by a gantry blade into a header card, body cards of ten cells, and a tail card. Stamps press a seal from each card and from each of sixteen branch cards in a tray. The seals fall into pockets along a trough, a comb checks every joint between them, and a press crimps a lead seal on the end of the chain.',
		body,
	});
	return { svg, P, cardWidth };
}

export const KEYS = [
	{ mark: 'raised', label: 'Cards: header, body chunks of ten, tail, and the sixteen branches' },
	{ mark: 'lit', label: 'The sealed proof, lit once it verifies' },
];

/** The one control: block size in transactions, in steps of one body chunk. */
export const controlHtml = (id = 'keth-size') =>
	`<div class="keth-control"><label class="keth-label" for="${id}">Block size</label><div class="keth-range"><input id="${id}" type="range" min="0" max="4" step="1" value="2" aria-valuetext="30 transactions"><div class="keth-ticks" aria-hidden="true"><span>10</span><span>20</span><span>30</span><span>40</span><span>50</span></div></div></div>`;
