/** One card's pose: translate in % of the card's own size, rotate in deg, rotateY (`ry`, deg, in perspective), scale, dim 0 to 1 (the opacity of the card's dim overlay, never a filter). */
export type Pose = { x: number; y: number; r: number; ry: number; s: number; d: number };

/** How a lifted card is held: `fan` spreads the hand sideways like playing cards; `pull` draws one out of the pile, up. */
export type Hold = 'fan' | 'pull';

export type DeckOptions = { spread: number; angle: number; lift: number; hold: Hold };

/** Fired on the Deck when the held card (the front card at rest) changes. A `Readout` follows it. */
export const ACTIVE_EVENT = 'deck:active';
export type DeckActive = { item: number; caption: Record<string, string> };

/** Opacity of a fully dimmed (receding) card's overlay. */
export const DIM = 0.07;

/** Percent of card height the pile is raised by before its knock on the table. */
export const KNOCK = 3;
/** How far a pulled card turns about its vertical axis, deg. */
const PULL_TURN = -14;

/** How far each card behind peeks out of the pile, % of card size (x right, y up). A pulled pile peeks more: the back book must be a target. */
const peek = (hold: Hold) => (hold === 'pull' ? { x: 12, y: 7 } : { x: 7, y: 4 });

/** Stacked: the front card readable, the next ones peeking up and right, slightly rotated. */
export function restPose(i: number, n: number, hold: Hold = 'fan'): Pose {
	const k = peek(hold);
	return {
		x: i * k.x - ((n - 1) * k.x) / 2,
		y: -i * k.y + ((n - 1) * k.y) / 2,
		r: i * 2.5,
		ry: 0,
		s: 1 - i * 0.03,
		d: 0,
	};
}

/**
 * The hand position under a pointer at (u, v), in % of the pile's front-card box. For a pulled pile this is
 * physical: the card whose resting box is hit first from the front wins, so the peeking strip of a back book
 * pulls that book; off the pile, the pointer's place along the pile's offset direction decides.
 */
export function pullZone(u: number, v: number, n: number): number {
	const k = peek('pull');
	for (let p = 0; p < n; p++) {
		const x = p * k.x - ((n - 1) * k.x) / 2;
		const y = -p * k.y + ((n - 1) * k.y) / 2;
		if (u >= x && u <= x + 100 && v >= y && v <= y + 100) return p;
	}
	// off the pile: project the pointer, from the front card's centre, onto the offset direction
	const du = u - 50 + ((n - 1) * k.x) / 2;
	const dv = v - 50 - ((n - 1) * k.y) / 2;
	const steps = (du * k.x - dv * k.y) / (k.x * k.x + k.y * k.y);
	return Math.min(n - 1, Math.max(0, Math.round(steps)));
}

/** The held pose of card `i` (hand position) while `active` is lifted. */
export function openPose(i: number, n: number, o: DeckOptions, active: number): Pose {
	if (o.hold === 'pull') {
		// the pile stays where it is; the active book comes up and out of it, turned to show its depth
		const rest = restPose(i, n, 'pull');
		if (i !== active) return { ...rest, d: 1 };
		return { ...rest, y: rest.y - o.lift, r: 0, ry: PULL_TURN, s: 1.03 };
	}
	// fanned like a hand (beUI `getPreviewTransform`): offset from the centre card drives x, rotation, arc and scale
	const offset = i - (n - 1) / 2;
	const arc = offset * offset * 2;
	if (i === active)
		return { x: offset * o.spread * 100, y: arc - o.lift, r: 0, ry: 0, s: 1.04, d: 0 };
	return {
		x: offset * o.spread * 100,
		y: arc,
		r: offset * o.angle,
		ry: 0,
		s: 0.96 - Math.abs(offset) * 0.01,
		d: 1,
	};
}

/** The CSS transform of a pose. Perspective and rotateY only while turned, so a flat card is not promoted to a 3D layer (blurry text). */
export const transformOf = (p: Omit<Pose, 'd'>) =>
	`translate(${p.x}%, ${p.y}%)${p.ry ? ` perspective(900px) rotateY(${p.ry}deg)` : ''} rotate(${p.r}deg) scale(${p.s})`;
