import { box, cyl, P, PLANE, pts } from '@/ds/components/objects/iso';
import { IsoGlyph, IsoPart } from '@/ds/components/objects/iso-glyph';

const plinth = box(-20, -16, 20, 16, 0, 4);
const vinyl = cyl(-4, 0, 4, 5.5, 12);
const base = cyl(15, -10, 4, 6.5, 2.6);
const pivot = P(15, -10, 7);
const weight = P(18.5, -15, 7);
const stylus = P(3, 5, 7);
const head = pts([2, 3.6, 7], [4.6, 5.4, 7], [3.6, 6.8, 7], [1, 5, 7]);
const light = P(-15, 11, 4);

/** Plinth, vinyl with a label tick, tonearm, one orange play light. The vinyl turns flat on hover. */
export function Turntable({ size }: { size?: number }) {
	return (
		<IsoGlyph size={size} frame={[0, -1.75, 60]} className="glyph-turntable">
			<polygon className="l" points={plinth.left} />
			<polygon className="r" points={plinth.right} />
			<polygon className="t" points={plinth.top} />
			<g transform={`translate(${light.join(' ')}) ${PLANE}`}>
				<circle className="light" r="1.5" />
			</g>
			<path className="l" d={vinyl.left} />
			<path className="r" d={vinyl.right} />
			<g transform={`translate(${vinyl.X} ${vinyl.Y}) ${PLANE}`}>
				<IsoPart>
					<circle className="r" r="12" />
					<circle className="groove" r="8.5" />
					<circle className="t" r="4.6" />
					<line x1="0" y1="0" x2="4.6" y2="0" />
				</IsoPart>
			</g>
			<path className="l" d={base.left} />
			<path className="r" d={base.right} />
			<ellipse className="t" cx={base.X} cy={base.Y} rx="2.6" ry="1.3" />
			<line className="arm" x1={weight[0]} y1={weight[1]} x2={pivot[0]} y2={pivot[1]} />
			<line className="arm" x1={pivot[0]} y1={pivot[1]} x2={stylus[0]} y2={stylus[1]} />
			<polygon className="t" points={head} />
		</IsoGlyph>
	);
}
