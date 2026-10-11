/** One pose of a `FanItem`: translate, rotate (deg) and scale. */
export type Pose = { x?: string; y?: string; r?: number; s?: number };

/** Inline custom properties for a pose, e.g. `--rest-x`. */
export function poseVars(name: string, p: Pose = {}): string {
	return `--${name}-x:${p.x ?? '0px'};--${name}-y:${p.y ?? '0px'};--${name}-r:${p.r ?? 0}deg;--${name}-s:${p.s ?? 1}`;
}
