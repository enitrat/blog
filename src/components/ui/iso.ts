/** Isometric projection shared by IsoGlyph objects: 45° azimuth, 30° elevation, x right-down, y left-down, z up. */
const K = Math.SQRT1_2;
const H = K / 2;
const Z = Math.sqrt(3) / 2;

type V = [number, number, number];
const n = (v: number) => +v.toFixed(2);

/** Plan point (x, y) at height z to screen. */
export const P = (x: number, y: number, z = 0): [number, number] => [
	n((x - y) * K),
	n((x + y) * H - z * Z),
];
export const pts = (...p: V[]) => p.map(([x, y, z]) => P(x, y, z).join(',')).join(' ');

/** The three visible faces of a box; light from the top left: left face is y1, right face is x1. */
export const box = (x0: number, y0: number, x1: number, y1: number, z0: number, z1: number) => ({
	top: pts([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]),
	left: pts([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]),
	right: pts([x1, y1, z1], [x1, y0, z1], [x1, y0, z0], [x1, y1, z0]),
});

/** Flat plan matrix: draw un-projected plan shapes (circles, rotations) inside it. */
export const PLANE = `matrix(${n(K)} ${n(H)} ${n(-K)} ${n(H)} 0 0)`;

/** A vertical cylinder: top ellipse centre, radii, and the lit left / shaded right side paths. */
export const cyl = (cx: number, cy: number, z0: number, z1: number, r: number) => {
	const [X, Y] = P(cx, cy, z1);
	const ry = n(r / 2);
	const h = n((z1 - z0) * Z);
	return {
		X,
		Y,
		ry,
		left: `M${n(X - r)},${Y}A${r},${ry} 0 0 0 ${X},${n(Y + ry)}L${X},${n(Y + ry + h)}A${r},${ry} 0 0 1 ${n(X - r)},${n(Y + h)}Z`,
		right: `M${X},${n(Y + ry)}A${r},${ry} 0 0 0 ${n(X + r)},${Y}L${n(X + r)},${n(Y + h)}A${r},${ry} 0 0 1 ${X},${n(Y + ry + h)}Z`,
	};
};
