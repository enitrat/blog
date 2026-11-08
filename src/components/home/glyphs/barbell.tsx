import { Fragment } from 'react';
import { P } from './iso';
import { IsoGlyph, IsoPart } from './iso-glyph';

const R = 8.5;
type Disc = [x0: number, x1: number, radius: number];
// Plates and collars in painter's order along x (far to near).
const far: Disc[] = [
	[-20.5, -18.5, 2.4],
	[-18.5, -16, 6],
	[-16, -13, R],
];
const near: Disc[] = [
	[13, 16, R],
	[16, 18.5, 6],
	[18.5, 20.5, 2.4],
];
const plane = (x: number) => {
	const [cx, cy] = P(x, 0, R);
	return `matrix(-0.7071 0.3536 0 -0.866 ${cx} ${cy})`;
};
const steps = (x0: number, x1: number) => {
	const out: number[] = [];
	for (let x = x0; x < x1; x += 0.5) out.push(x);
	return out;
};
const bar = (x0: number, x1: number) => ({ a: P(x0, 0, R), b: P(x1, 0, R) });
const mid = bar(-13, 13);
const tip = bar(20.5, 26);

function Bar({ seg }: { seg: ReturnType<typeof bar> }) {
	return (
		<>
			<line style={{ strokeWidth: 3.2 }} x1={seg.a[0]} y1={seg.a[1]} x2={seg.b[0]} y2={seg.b[1]} />
			<line
				className="stroke-white! dark:stroke-iron-600!"
				style={{ strokeWidth: 1.4 }}
				x1={seg.a[0]}
				y1={seg.a[1]}
				x2={seg.b[0]}
				y2={seg.b[1]}
			/>
		</>
	);
}

/** A bar loaded with two plates a side and collars, lying on the floor along x. It dips on hover. */
export function Barbell({ size }: { size?: number }) {
	return (
		<IsoGlyph size={size} frame={[0.55, -6.6, 42]}>
			<g transform="translate(0 3)">
				<IsoPart pose="motion-safe:group-hover/glyph:translate-y-[1.5px]">
					{[far, near].map((group, i) => (
						<Fragment key={i === 0 ? 'far' : 'near'}>
							{i === 1 && <Bar seg={mid} />}
							{group.map(([x0, x1, r]) => (
								<Fragment key={x0}>
									{steps(x0, x1).map((x) => (
										<circle key={x} className="l" transform={plane(x)} r={r} />
									))}
									<circle className="r" transform={plane(x1)} r={r} />
								</Fragment>
							))}
						</Fragment>
					))}
					<Bar seg={tip} />
				</IsoPart>
			</g>
		</IsoGlyph>
	);
}
