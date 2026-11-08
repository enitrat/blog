import { box, pts } from './iso';
import { IsoGlyph, IsoPart } from './iso-glyph';

const HAIR = { fill: 'none', opacity: 0.5 };

const foot = box(-15, -11, 15, 11, 0, 1.2);
const pages = box(-14, -10, 14, 10.4, 1.2, 5.8);
const spine = box(-15, 10.4, 15, 11, 0, 7);
const cover = box(-15, -11, 15, 11, 5.8, 7);
const sheets = [2.2, 3.4, 4.6].map((z) => pts([14, 10.4, z], [14, -10, z]));
const band = [-9, 9].map((x) => pts([x, 11, 1.4], [x, 11, 5.6]));
const title = [pts([-9, -3, 7], [5, -3, 7]), pts([-9, -0.4, 7], [-1, -0.4, 7])];
const frame = pts(
	[-12.5, -8.5, 7],
	[12.5, -8.5, 7],
	[12.5, 8.5, 7],
	[-12.5, 8.5, 7],
	[-12.5, -8.5, 7],
);

/** A closed hardback lying flat; the spine faces left. The top board lifts a hair on hover. */
export function Book({ size }: { size?: number }) {
	return (
		<IsoGlyph size={size} frame={[0, -3.9, 43]}>
			<polygon className="l" points={foot.left} />
			<polygon className="r" points={foot.right} />
			<polygon className="r" points={pages.right} />
			<polygon className="t" points={pages.top} />
			{sheets.map((s) => (
				<polyline key={s} style={HAIR} points={s} />
			))}
			<polygon className="l" points={spine.left} />
			<polygon className="r" points={spine.right} />
			{band.map((b) => (
				<polyline key={b} style={HAIR} points={b} />
			))}
			<IsoPart pose="motion-safe:group-hover/glyph:-translate-y-[1.6px]">
				<polygon className="l" points={cover.left} />
				<polygon className="r" points={cover.right} />
				<polygon className="t" points={cover.top} />
				<polyline style={HAIR} points={frame} />
				{title.map((t) => (
					<polyline key={t} points={t} />
				))}
			</IsoPart>
		</IsoGlyph>
	);
}
