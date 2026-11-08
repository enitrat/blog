import 'server-only';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ImageResponse } from 'next/og';
import sharp from 'sharp';

export const OG_SIZE = { width: 1200, height: 630 };

/** The room palette, converted from OKLCH to sRGB hex because Satori reads no OKLCH. */
const PAPER = '#f8f5f1'; // oklch(0.972 0.006 85)
const INK = '#1c1612'; // oklch(0.205 0.012 50)
const MUTED = '#49403c'; // oklch(0.38 0.014 50)
const ACCENT = '#9e2c2c'; // oklch(0.47 0.15 25)
const CARD = '#fefdfb'; // oklch(0.995 0.003 85)

const PLATE = { width: 440, height: 330 };

const read = (file: string) => readFileSync(join(/*turbopackIgnore: true*/ process.cwd(), file));

const fonts = () => [
	{
		name: 'Instrument Serif',
		data: read('src/assets/fonts/InstrumentSerif-Regular.ttf'),
		weight: 400 as const,
	},
	{
		name: 'Geist',
		data: read('node_modules/geist/dist/fonts/geist-sans/Geist-Regular.ttf'),
		weight: 400 as const,
	},
	{
		name: 'Geist',
		data: read('node_modules/geist/dist/fonts/geist-sans/Geist-Medium.ttf'),
		weight: 500 as const,
	},
];

/**
 * A checked-in image cropped to the plate at twice its size, as a JPEG data URL. Satori hands the
 * whole data URL to its SVG parser, which fails on the multi-megabyte banners.
 */
export async function plateUrl(file: string): Promise<string> {
	const jpeg = await sharp(read(file))
		.resize(PLATE.width * 2, PLATE.height * 2, { fit: 'cover' })
		.flatten({ background: CARD })
		.jpeg({ quality: 85 })
		.toBuffer();
	return `data:image/jpeg;base64,${jpeg.toString('base64')}`;
}

/** The card: paper ground, an oxblood rule, text on the left and an optional framed plate on the right. */
export function ogImage({
	title,
	titleSize,
	kicker,
	plate,
	footer,
}: {
	title: string;
	titleSize: number;
	kicker: string;
	plate?: string;
	footer: string;
}): ImageResponse {
	return new ImageResponse(
		<div
			style={{
				width: '100%',
				height: '100%',
				display: 'flex',
				background: PAPER,
				color: INK,
				padding: 72,
				fontFamily: 'Geist',
			}}
		>
			<div
				style={{
					display: 'flex',
					flexDirection: 'column',
					flex: 1,
					justifyContent: 'space-between',
				}}
			>
				<div style={{ width: 72, height: 6, background: ACCENT }} />
				<div style={{ display: 'flex', flexDirection: 'column' }}>
					<div
						style={{
							display: 'flex',
							fontFamily: 'Instrument Serif',
							fontSize: titleSize,
							lineHeight: 1.04,
							letterSpacing: -1.5,
							paddingRight: plate ? 40 : 0,
						}}
					>
						{title}
					</div>
					<div
						style={{
							display: 'flex',
							fontSize: 30,
							lineHeight: 1.35,
							color: MUTED,
							marginTop: 28,
							paddingRight: plate ? 40 : 120,
						}}
					>
						{kicker}
					</div>
				</div>
				<div style={{ display: 'flex', fontSize: 26, fontWeight: 500, color: ACCENT }}>
					{footer}
				</div>
			</div>
			{plate ? (
				<div style={{ display: 'flex', alignItems: 'center' }}>
					<div
						style={{
							display: 'flex',
							background: CARD,
							padding: 14,
							border: `2px solid ${INK}`,
							borderRadius: 6,
						}}
					>
						{/* biome-ignore lint/performance/noImgElement: Satori draws plain img elements */}
						<img
							src={plate}
							width={PLATE.width}
							height={PLATE.height}
							style={{ objectFit: 'cover', borderRadius: 2 }}
							alt=""
						/>
					</div>
				</div>
			) : null}
		</div>,
		{ ...OG_SIZE, fonts: fonts() },
	);
}
