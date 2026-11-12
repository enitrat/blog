import 'server-only';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ImageResponse } from 'next/og';
import sharp from 'sharp';
import { familyOf } from './type-families';

export const OG_SIZE = { width: 1200, height: 630 };

/** The palette, converted from OKLCH to sRGB hex because Satori reads no OKLCH. */
const PAPER = '#f3f6fa'; // neutral-50, oklch(0.972 0.006 250)
const INK = '#0d131a'; // neutral-950, oklch(0.185 0.016 250)
const MUTED = '#484e54'; // neutral-700, oklch(0.42 0.013 250)
const ACCENT = '#2f62ac'; // accent-600, oklch(0.5 0.13 258)
const CARD = '#fafdff'; // neutral-0, oklch(0.993 0.006 250)

const PLATE = { width: 440, height: 330 };

const read = (file: string) => readFileSync(join(/*turbopackIgnore: true*/ process.cwd(), file));

/**
 * Satori reads neither CSS nor woff2, so the card carries static TTF cuts of the sans family and
 * restates its roles by hand: title as headline (600, −0.025em), kicker as lead, footer as control.
 */
const SANS = familyOf('--font-sans').name;
const fonts = () =>
	(
		[
			[400, 'Regular'],
			[500, 'Medium'],
			[600, 'SemiBold'],
		] as const
	).map(([weight, cut]) => ({
		name: SANS,
		data: read(`src/assets/fonts/${SANS.replaceAll(' ', '')}-${cut}.ttf`),
		weight,
	}));

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

/** The card: paper ground, an ink rule, text on the left and an optional framed plate on the right. */
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
				fontFamily: SANS,
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
							fontSize: titleSize,
							fontWeight: 600,
							lineHeight: 1.06,
							letterSpacing: titleSize * -0.025,
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
