/** One card's pose: translate in % of the card's own size, rotate in deg, scale, dim 0 to 1 (an overlay's opacity, never a filter). */
export type Pose = { x: number; y: number; r: number; s: number; d: number };

export type DeckOptions = { spread: number; angle: number; lift: number };

/** Stacked: the front card readable, the next ones peeking up and right, slightly rotated. */
export function restPose(i: number, n: number): Pose {
	return { x: i * 7 - (n - 1) * 3.5, y: -i * 4 + (n - 1) * 2, r: i * 2.5, s: 1 - i * 0.03, d: 0 };
}

/** Fanned like a hand (beUI `getPreviewTransform`): offset from the centre card drives x, rotation, arc and scale. */
export function openPose(i: number, n: number, o: DeckOptions, active: number): Pose {
	const offset = i - (n - 1) / 2;
	const arc = offset * offset * 2;
	if (i === active) return { x: offset * o.spread * 100, y: arc - o.lift, r: 0, s: 1.04, d: 0 };
	return {
		x: offset * o.spread * 100,
		y: arc,
		r: offset * o.angle,
		s: 0.96 - Math.abs(offset) * 0.01,
		d: 1,
	};
}

export const transformOf = (p: Pose) =>
	`translate(${p.x}%, ${p.y}%) rotate(${p.r}deg) scale(${p.s})`;
