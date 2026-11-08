import { box, P } from './iso';
import { IsoGlyph, IsoPart } from './iso-glyph';

const b = box(-11, -15, 11, 15, 0, 3);
const top = P(0, 0, 3);

/** A small tag with "25" printed on its top face, in the iso plane. It lifts on hover. */
export function AgePlate({ size }: { size?: number }) {
	return (
		<IsoGlyph size={size} frame={[0, -2.2, 43]}>
			<g transform="scale(1.2)">
				<IsoPart pose="motion-safe:group-hover/glyph:-translate-y-[1.5px]">
					<polygon className="l" points={b.left} />
					<polygon className="r" points={b.right} />
					<polygon className="t" points={b.top} />
					<g transform={`translate(${top.join(' ')}) matrix(0.7071 -0.3536 0.7071 0.3536 0 0)`}>
						<circle className="r" cx="-12" cy="-7.5" r="1.3" />
						<g
							transform="translate(-6 -6.5) scale(1.3)"
							style={{ fill: 'none', strokeWidth: 1.5, vectorEffect: 'none' }}
						>
							<path d="M0.5 2.6C0.5 0.4 5.6 0.4 5.6 3C5.6 5.6 0.5 7.4 0.5 9.6H5.6" />
							<path
								transform="translate(8.5 0)"
								d="M5.3 0.5H1.2L0.8 4.4C3 3.4 5.7 4 5.7 6.7C5.7 9.7 2 10.3 0.4 8.8"
							/>
						</g>
					</g>
				</IsoPart>
			</g>
		</IsoGlyph>
	);
}
