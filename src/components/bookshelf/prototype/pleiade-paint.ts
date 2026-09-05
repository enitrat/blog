import type { PleiadeColor } from '../../../utils/pleiade';

export const PLEIADE_HEX: Record<PleiadeColor, string> = {
	green: '#416653',
	violet: '#574363',
	corinthe: '#70483f',
	red: '#813b34',
	blue: '#315a78',
	emerald: '#2f6652',
	havane: '#76533e',
	grey: '#62615d',
};

const GOLD = '#f0d98d';
const GOLD_EDGE = 'rgba(225, 195, 117, 0.58)';
const GOLD_PANEL = 'rgba(225, 195, 117, 0.78)';

export type PaintedBook = {
	author: string;
	title: string;
	color: string;
};

export function shortTitle(title: string) {
	return title.replace(/\s*\(.*/, '');
}

function hexToRgb(hex: string): [number, number, number] {
	const n = Number.parseInt(hex.slice(1), 16);
	return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mixHex(a: string, b: string, t: number) {
	const [ar, ag, ab] = hexToRgb(a);
	const [br, bg, bb] = hexToRgb(b);
	const r = Math.round(ar + (br - ar) * t);
	const g = Math.round(ag + (bg - ag) * t);
	const bch = Math.round(ab + (bb - ab) * t);
	return `#${[r, g, bch].map((value) => value.toString(16).padStart(2, '0')).join('')}`;
}

function leatherTones(color: string) {
	return {
		leather: color,
		dark: mixHex(color, '#17120f', 0.32),
		light: mixHex(color, '#ffffff', 0.16),
	};
}

function wrapLines(ctx: CanvasRenderingContext2D, text: string, width: number) {
	const words = text.split(/\s+/);
	const lines: string[] = [];
	let current = '';
	for (const word of words) {
		const next = current ? `${current} ${word}` : word;
		if (ctx.measureText(next).width > width && current) {
			lines.push(current);
			current = word;
		} else current = next;
	}
	if (current) lines.push(current);
	return lines;
}

function fillLines(
	ctx: CanvasRenderingContext2D,
	lines: string[],
	x: number,
	y: number,
	width: number,
	line: number,
) {
	for (const [index, value] of lines.entries()) {
		ctx.fillText(value, x, y + index * line, width);
	}
}

function fitLines(
	ctx: CanvasRenderingContext2D,
	text: string,
	width: number,
	height: number,
	start: number,
	min: number,
	leading: number,
) {
	for (let size = start; size >= min; size -= 1) {
		ctx.font = `620 ${size}px "Literata Variable", Georgia`;
		const lines = wrapLines(ctx, text, width);
		const line = size * leading;
		if (lines.length * line <= height) return { lines, size, line };
	}
	ctx.font = `620 ${min}px "Literata Variable", Georgia`;
	return { lines: wrapLines(ctx, text, width), size: min, line: min * leading };
}

function grain(ctx: CanvasRenderingContext2D, width: number, height: number, seed: number) {
	const count = Math.floor((width * height) / 90);
	for (let i = 0; i < count; i++) {
		const n = Math.sin((i + 1) * 127.1 + seed) * 43758.5453;
		const noise = n - Math.floor(n);
		const n2 = Math.sin((i + 1) * 269.3 + seed) * 19672.84;
		const noise2 = n2 - Math.floor(n2);
		ctx.fillStyle =
			noise > 0.5
				? `rgba(255,255,255,${0.015 + noise * 0.03})`
				: `rgba(0,0,0,${0.02 + noise * 0.04})`;
		ctx.fillRect(noise * width, noise2 * height, 1.2, 1.2);
	}
}

function cylinderShade(ctx: CanvasRenderingContext2D, width: number, height: number, dark: string) {
	const shade = ctx.createLinearGradient(0, 0, width, 0);
	shade.addColorStop(0, `${dark}a8`);
	shade.addColorStop(0.28, 'rgba(255,255,255,0)');
	shade.addColorStop(0.58, 'rgba(255,255,255,0.14)');
	shade.addColorStop(1, `${dark}a8`);
	ctx.fillStyle = shade;
	ctx.fillRect(0, 0, width, height);
}

function filets(ctx: CanvasRenderingContext2D, width: number, height: number) {
	const cycle = height / 68;
	const gold = Math.max(1.5, cycle * 0.32);
	ctx.fillStyle = 'rgba(240, 217, 141, 0.82)';
	for (let y = 0; y < height; y += cycle) ctx.fillRect(0, y + cycle - gold, width, gold);
}

export function paintSpine(ctx: CanvasRenderingContext2D, book: PaintedBook) {
	const { width, height } = ctx.canvas;
	const tone = leatherTones(book.color);
	ctx.fillStyle = tone.leather;
	ctx.fillRect(0, 0, width, height);
	filets(ctx, width, height);
	cylinderShade(ctx, width, height, tone.dark);
	grain(ctx, width, height, book.title.length);

	const insetX = Math.max(3, width * 0.06);
	const insetY = Math.max(4, height * 0.02);
	ctx.strokeStyle = GOLD_EDGE;
	ctx.lineWidth = Math.max(1, width * 0.012);
	ctx.strokeRect(insetX, insetY, width - insetX * 2, height - insetY * 2);

	const panelTop = height * 0.27;
	const panelHeight = height * 0.44;
	const panelX = Math.max(3, width * 0.05);
	const panelW = width - panelX * 2;
	const panel = ctx.createLinearGradient(panelX, 0, panelX + panelW, 0);
	panel.addColorStop(0, tone.dark);
	panel.addColorStop(0.24, tone.leather);
	panel.addColorStop(0.58, tone.light);
	panel.addColorStop(1, tone.dark);
	ctx.fillStyle = panel;
	ctx.fillRect(panelX, panelTop, panelW, panelHeight);
	ctx.strokeStyle = GOLD_PANEL;
	ctx.lineWidth = Math.max(1, height * 0.002);
	ctx.beginPath();
	ctx.moveTo(panelX, panelTop);
	ctx.lineTo(panelX + panelW, panelTop);
	ctx.moveTo(panelX, panelTop + panelHeight);
	ctx.lineTo(panelX + panelW, panelTop + panelHeight);
	ctx.stroke();

	ctx.fillStyle = GOLD;
	ctx.textAlign = 'center';
	ctx.textBaseline = 'top';
	const textX = width / 2;
	const textW = panelW - width * 0.08;
	const author = fitLines(
		ctx,
		book.author.toUpperCase(),
		textW,
		panelHeight * 0.32,
		Math.round(width / 8.2),
		Math.max(9, Math.round(width / 14)),
		1.12,
	);
	const authorY = panelTop + panelHeight * 0.1;
	ctx.font = `720 ${author.size}px "Literata Variable", Georgia`;
	fillLines(ctx, author.lines, textX, authorY, textW, author.line);

	const titleTop = authorY + author.lines.length * author.line + panelHeight * 0.06;
	const titleRoom = panelTop + panelHeight * 0.78 - titleTop;
	const title = fitLines(
		ctx,
		shortTitle(book.title).toUpperCase(),
		textW,
		titleRoom,
		Math.round(width / 7.8),
		Math.max(9, Math.round(width / 13)),
		1.16,
	);
	ctx.font = `620 ${title.size}px "Literata Variable", Georgia`;
	fillLines(ctx, title.lines, textX, titleTop, textW, title.line);

	const markY = panelTop + panelHeight * 0.88;
	const markR = Math.max(1.6, width * 0.035);
	const gap = markR * 3.2;
	ctx.strokeStyle = GOLD;
	ctx.lineWidth = Math.max(1, width * 0.02);
	for (const x of [textX - gap / 2, textX + gap / 2]) {
		ctx.beginPath();
		ctx.arc(x, markY, markR, 0, Math.PI * 2);
		ctx.stroke();
	}
}

export function paintCover(ctx: CanvasRenderingContext2D, book: PaintedBook) {
	const { width, height } = ctx.canvas;
	const tone = leatherTones(book.color);
	ctx.fillStyle = tone.leather;
	ctx.fillRect(0, 0, width, height);
	const light = ctx.createRadialGradient(
		width * 0.22,
		height * 0.18,
		0,
		width * 0.3,
		height * 0.2,
		height,
	);
	light.addColorStop(0, 'rgba(255,255,255,0.1)');
	light.addColorStop(1, 'rgba(0,0,0,0)');
	ctx.fillStyle = light;
	ctx.fillRect(0, 0, width, height);
	const shade = ctx.createLinearGradient(0, 0, width, 0);
	shade.addColorStop(0, `${tone.dark}55`);
	shade.addColorStop(0.18, 'rgba(255,255,255,0)');
	shade.addColorStop(0.92, 'rgba(255,255,255,0)');
	shade.addColorStop(1, `${tone.dark}66`);
	ctx.fillStyle = shade;
	ctx.fillRect(0, 0, width, height);
	grain(ctx, width, height, book.author.length + 11);

	const outer = Math.max(10, width * 0.045);
	const inner = outer + Math.max(5, width * 0.018);
	ctx.strokeStyle = GOLD_PANEL;
	ctx.lineWidth = Math.max(1.4, width * 0.008);
	ctx.strokeRect(outer, outer, width - outer * 2, height - outer * 2);
	ctx.strokeStyle = GOLD_EDGE;
	ctx.lineWidth = Math.max(1, width * 0.004);
	ctx.strokeRect(inner, inner, width - inner * 2, height - inner * 2);

	ctx.fillStyle = GOLD;
	ctx.textAlign = 'center';
	ctx.textBaseline = 'top';
	const textW = width - inner * 2 - width * 0.08;
	const author = fitLines(
		ctx,
		book.author.toUpperCase(),
		textW,
		height * 0.14,
		Math.round(width * 0.048),
		Math.round(width * 0.032),
		1.2,
	);
	ctx.font = `620 ${author.size}px "Literata Variable", Georgia`;
	fillLines(ctx, author.lines, width / 2, height * 0.22, textW, author.line);

	const title = fitLines(
		ctx,
		book.title,
		textW,
		height * 0.28,
		Math.round(width * 0.09),
		Math.round(width * 0.048),
		1.14,
	);
	ctx.font = `500 ${title.size}px "Literata Variable", Georgia`;
	fillLines(ctx, title.lines, width / 2, height * 0.38, textW, title.line);

	ctx.font = `500 ${Math.round(width * 0.038)}px "Literata Variable", Georgia`;
	ctx.fillText('Bibliothèque personnelle', width / 2, height * 0.78, textW);
	const diamond = Math.max(4, width * 0.012);
	ctx.translate(width / 2, height * 0.86);
	ctx.rotate(Math.PI / 4);
	ctx.strokeStyle = GOLD;
	ctx.lineWidth = Math.max(1, width * 0.004);
	ctx.strokeRect(-diamond, -diamond, diamond * 2, diamond * 2);
	ctx.setTransform(1, 0, 0, 1, 0, 0);
}

export function paintBackCover(ctx: CanvasRenderingContext2D, color: string) {
	const { width, height } = ctx.canvas;
	const tone = leatherTones(color);
	ctx.fillStyle = tone.leather;
	ctx.fillRect(0, 0, width, height);
	const shade = ctx.createLinearGradient(0, 0, width, 0);
	shade.addColorStop(0, `${tone.dark}66`);
	shade.addColorStop(0.5, 'rgba(255,255,255,0.06)');
	shade.addColorStop(1, `${tone.dark}88`);
	ctx.fillStyle = shade;
	ctx.fillRect(0, 0, width, height);
	grain(ctx, width, height, 3);
	const outer = Math.max(10, width * 0.045);
	ctx.strokeStyle = GOLD_EDGE;
	ctx.lineWidth = Math.max(1.2, width * 0.006);
	ctx.strokeRect(outer, outer, width - outer * 2, height - outer * 2);
}

export function paintGilt(ctx: CanvasRenderingContext2D) {
	const { width, height } = ctx.canvas;
	const wash = ctx.createLinearGradient(0, 0, 0, height);
	wash.addColorStop(0, '#d7c48a');
	wash.addColorStop(0.45, '#f3e4b4');
	wash.addColorStop(1, '#c4ad74');
	ctx.fillStyle = wash;
	ctx.fillRect(0, 0, width, height);
	ctx.fillStyle = '#efe6c4';
	for (let y = 1; y < height; y += 2) {
		ctx.globalAlpha = 0.35 + ((y * 13) % 7) * 0.04;
		ctx.fillRect(0, y, width, 1);
	}
	ctx.globalAlpha = 1;
	const bind = ctx.createLinearGradient(0, 0, width, 0);
	bind.addColorStop(0, 'rgba(40, 24, 10, 0.35)');
	bind.addColorStop(0.12, 'rgba(40, 24, 10, 0)');
	ctx.fillStyle = bind;
	ctx.fillRect(0, 0, width, height);
}
