// Keth bench: the pure model. Everything the figure moves or reads comes from here.
// Sourced facts only (kkrt-labs/keth): body chunks hold 10 txs by default, mpt_diff has 16
// chunks (one per branch of the state trie), and the aggregator chains init, N bodies, teardown
// and the 16 mpt_diff outputs. No timings of the real prover appear anywhere.

export const CHUNK = 10;
export const BRANCHES = 16;
export const SIZES = [10, 20, 30, 40, 50];

export const bodiesOf = (txs) => Math.ceil(txs / CHUNK);
/** Segments the aggregator chains: init, bodies, teardown, 16 mpt_diff. */
export const chainOf = (txs) => bodiesOf(txs) + 2 + BRANCHES;

/** Bench layout in world units (x along the bench, y across; larger x and y face the viewer). */
export const L = {
	X0: -218, // left end of the block strip
	INIT_W: 34,
	BODY_W: 50,
	TEAR_W: 34,
	GAP: 9, // opening of a cut
	CARD_Y: 63, // centre line of the card lane
	CARD_D: 34,
	CARD_TOP: 6.5,
	TRAY_Y: -71,
	SLOT: 12,
	SLOT0: 6, // x of the first mpt slot centre
	TROUGH_Y: -2,
	POCKET: 14, // pitch of the seal pockets
	XR: 168, // x of pocket 0 (right end of the trough)
	POCKETS: 23, // BODIES max 5 + 2 + 16
	PROOF_X: 224,
	BLADE_DX: 7.6, // blade hangs on the near face of the beam
	STAMP_DX: 18, // stamps hang in front of it
	PARK: -226,
	LIFT_BLADE: 8, // how far each tool rests above its working height
	LIFT_STAMP: 10,
};

export const COMB_REST = 14;
export const RAM_REST = 12;

export const cardCount = (txs) => bodiesOf(txs) + 2;
const joinedX = (txs, i) => {
	const B = bodiesOf(txs);
	if (i === 0) return L.X0;
	if (i <= B) return L.X0 + L.INIT_W + L.BODY_W * (i - 1);
	return L.X0 + L.INIT_W + L.BODY_W * B;
};
export const cardWidth = (txs, i) =>
	i === 0 ? L.INIT_W : i === bodiesOf(txs) + 1 ? L.TEAR_W : L.BODY_W;
export const cardX = (txs, i, cut = 1) => joinedX(txs, i) + L.GAP * i * cut;
export const slotX = (j) => L.SLOT0 + L.SLOT * j;
export const pocketX = (p) => L.XR - L.POCKET * p;
/** Pocket of chain index k (init 0, bodies, teardown, then branches 0 to 15). */
export const pocketOf = (txs, k) => chainOf(txs) - 1 - k;

const ramp = (t, a, b) => {
	if (t <= a) return 0;
	if (t >= b) return 1;
	const u = (t - a) / (b - a);
	return u * u * (3 - 2 * u);
};
/** 0 to 1 and back: down over [a, b], held to c, up over [c, d]. */
const pulse = (t, a, b, c, d) => ramp(t, a, b) * (1 - ramp(t, c, d));

/** Build the timeline for a block of `txs` transactions. Pure: state(t) is a function of t only. */
export function timeline(txs) {
	const B = bodiesOf(txs);
	const K = B + 2;
	const total = chainOf(txs);
	const gx = [[0, L.PARK]]; // [time, beam x] keyframes
	let t = 0.7;
	const cuts = []; // seam s: blade down, open
	for (let s = 0; s <= B; s++) {
		const seamX = L.X0 + L.INIT_W + L.BODY_W * s + L.GAP * s;
		const beamX = seamX - L.BLADE_DX;
		const prev = gx[gx.length - 1];
		const travel = Math.max(0.45, Math.abs(beamX - prev[1]) / 230);
		gx.push([t + travel, beamX]);
		t += travel;
		cuts.push({
			down: [t, t + 0.18],
			hold: t + 0.24,
			up: [t + 0.24, t + 0.42],
			open: [t + 0.18, t + 0.5],
		});
		t += 0.5;
		gx.push([t, beamX]);
	}
	const cutEnd = t;
	// stamp stops, right to left; a card and a slot closer than 5 share one stop
	const stops = [];
	for (let i = 0; i < K; i++) {
		const x = cardX(txs, i) + cardWidth(txs, i) / 2;
		stops.push({ x, card: i, slot: -1 });
	}
	for (let j = 0; j < BRANCHES; j++) {
		const x = slotX(j);
		const near = stops.find((s) => s.slot < 0 && Math.abs(s.x - x) < 5);
		if (near) near.slot = j;
		else stops.push({ x, card: -1, slot: j });
	}
	stops.sort((a, b) => b.x - a.x);
	t += 0.3;
	const hits = []; // {k, t: contact time, src: [x, y, z]}
	const stampF = [];
	const stampB = [];
	for (const s of stops) {
		const beamX = s.x - L.STAMP_DX;
		const prev = gx[gx.length - 1];
		const travel = Math.max(0.1, Math.abs(beamX - prev[1]) / 380);
		gx.push([t + travel, beamX]);
		t += travel;
		const dur = s.card >= 0 ? 0.36 : 0.18;
		const down = [t, t + dur * 0.35];
		const up = [t + dur * 0.5, t + dur * 0.85];
		if (s.card >= 0) {
			stampF.push([...down, up[0], up[1]]);
			hits.push({ k: s.card, t: down[1], src: [s.x, L.CARD_Y, L.CARD_TOP], card: s.card });
		}
		if (s.slot >= 0) {
			stampB.push([...down, up[0], up[1]]);
			hits.push({
				k: K + s.slot,
				t: down[1],
				src: [slotX(s.slot), L.TRAY_Y, L.CARD_TOP],
				slot: s.slot,
			});
		}
		t += dur;
		gx.push([t, beamX]);
	}
	const FLY = 0.6;
	for (const h of hits) {
		h.fly = [h.t + 0.1, h.t + 0.1 + FLY];
		h.dst = [pocketX(pocketOf(txs, h.k)), L.TROUGH_Y, 3.4];
	}
	const landed = Math.max(...hits.map((h) => h.fly[1]));
	gx.push([landed + 0.4, L.PARK + 20]);
	const combT = landed + 0.45;
	const comb = [combT, combT + 0.6, combT + 1.1, combT + 1.7];
	const keysT = [combT + 0.65, combT + 1.05];
	const pressT = [combT + 1.8, combT + 2.25, combT + 2.4, combT + 2.85];
	const proofT = [combT + 2.2, combT + 2.4];
	const verifyT = [combT + 2.9, combT + 3.3];
	const end = combT + 4.8;
	const fadeOut = [end, end + 0.5];
	const stageAt = (time) => {
		if (time < 0.7) return 'block';
		if (time < cutEnd) {
			const n = cuts.filter((c) => time >= c.hold).length;
			return `cut ${n} of ${B + 1}`;
		}
		if (time < landed + 0.2) return `run ${hits.filter((h) => time >= h.t).length} of ${total}`;
		if (time < pressT[0]) return `chain ${total}`;
		if (time < verifyT[0]) return 'prove';
		return 'verify';
	};
	const gantryAt = (time) => {
		if (time <= gx[0][0]) return gx[0][1];
		for (let i = 1; i < gx.length; i++) {
			if (time <= gx[i][0]) {
				const [t0, x0] = gx[i - 1];
				const [t1, x1] = gx[i];
				return x0 + (x1 - x0) * ramp(time, t0, t1);
			}
		}
		return gx[gx.length - 1][1];
	};
	/** Where a narrow screen should look, 0 (left end of the bench) to 1 (the press): the gantry, then the press. */
	const focusAt = (time) => {
		const u = Math.min(1, Math.max(0, (gantryAt(time) - L.PARK) / 400));
		return u + (1 - u) * ramp(time, landed + 0.2, landed + 1.2);
	};
	const pulses = (list, time) => list.reduce((m, p) => Math.max(m, pulse(time, ...p)), 0);
	return {
		B,
		K,
		total,
		duration: end + 0.5,
		cutEnd,
		stageAt,
		/** Everything the page draws at time t. All values 0..1 unless noted. */
		at(time) {
			const open = cuts.map((c) => ramp(time, ...c.open));
			const cardShift = (i) => L.GAP * open.slice(0, i).reduce((a, b) => a + b, 0);
			return {
				gantry: gantryAt(time),
				blade: Math.max(0, ...cuts.map((c) => pulse(time, c.down[0], c.down[1], c.up[0], c.up[1]))),
				stampF: pulses(stampF, time),
				stampB: pulses(stampB, time),
				cardShift: Array.from({ length: K }, (_, i) => cardShift(i)),
				stamped: Array.from({ length: K }, (_, i) => {
					const h = hits.find((x) => x.card === i);
					return h ? ramp(time, h.t, h.t + 0.15) : 0;
				}),
				slotStamped: Array.from({ length: BRANCHES }, (_, j) => {
					const h = hits.find((x) => x.slot === j);
					return h ? ramp(time, h.t, h.t + 0.15) : 0;
				}),
				tokens: hits.map((h) => {
					const f = ramp(time, ...h.fly);
					const seen = ramp(time, h.t, h.t + 0.1);
					const arc = Math.sin(Math.PI * f) * 12;
					return {
						k: h.k,
						alpha: seen,
						pos: [
							h.src[0] + (h.dst[0] - h.src[0]) * f,
							h.src[1] + (h.dst[1] - h.src[1]) * f,
							h.src[2] + 1 + (h.dst[2] - h.src[2] - 1) * f + arc,
						],
					};
				}),
				comb: pulse(time, comb[0], comb[1], comb[2], comb[3]),
				keys: ramp(time, ...keysT),
				press: pulse(time, ...pressT),
				proof: ramp(time, ...proofT),
				verified: ramp(time, ...verifyT),
				focus: focusAt(time),
				fade: ramp(time, 0, 0.5) * (1 - ramp(time, ...fadeOut)),
				stage: stageAt(time),
			};
		},
	};
}

export function readoutOf(txs, stage) {
	const B = bodiesOf(txs);
	return `${txs} txs · ${B} body · ${stage}`;
}

/** Self-check: node model.mjs */
// biome-ignore lint/complexity/useOptionalChain: `process` is undeclared in the browser
if (typeof process !== 'undefined' && process.argv[1]?.endsWith('model.mjs')) {
	const assert = (c, m) => {
		if (!c) throw new Error(m);
	};
	for (const n of SIZES) {
		const tl = timeline(n);
		assert(tl.B === n / 10, 'bodies');
		assert(tl.total === n / 10 + 18, 'chain');
		const end = tl.at(tl.duration - 0.6);
		assert(end.verified === 1 && end.proof === 1, 'verified at end');
		assert(end.tokens.length === tl.total, 'one seal per segment');
		const pockets = new Set(end.tokens.map((x) => pocketOf(n, x.k)));
		assert(pockets.size === tl.total && Math.max(...pockets) === tl.total - 1, 'pockets unique');
		assert(
			end.tokens.every((x) => x.pos[0] === pocketX(pocketOf(n, x.k))),
			'tokens land in pockets',
		);
	}
	console.log('model ok');
}
