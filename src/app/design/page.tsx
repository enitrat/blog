import type { ReactNode } from 'react';
import { TypeSpec } from '@/components/design/type-spec';
import { StickerMark, Wordmark } from '@/components/site/wordmark';
import {
	Accordion,
	AccordionItem,
	AccordionPanel,
	AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Eyebrow } from '@/components/ui/eyebrow';
import { Page, PageHeader } from '@/components/ui/page';
import { StatusDot } from '@/components/ui/status-dot';
import { pageMeta } from '@/lib/meta';
import { TYPE_FAMILIES } from '@/lib/type-families';
import { cn } from '@/lib/utils';

export const metadata = pageMeta({
	title: 'Design',
	description: 'The primitives this site is built from: mark, type, colour, space and depth.',
	path: '/design/',
});

const STEPS = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900'];

/** The palettes of globals.css, read by variable. This page is the one place that shows them. */
const PALETTES = [
	{
		name: 'neutral',
		job: 'Surfaces and text',
		steps: ['0', '50', '100', '200', '300', '400', '500', '600', '700', '800', '850', '900', '950'],
	},
	{ name: 'accent', job: 'The brand colour, day and night', steps: STEPS },
	{ name: 'positive', job: 'Live, on, now', steps: STEPS },
];

/** Colour roles, each with its utility written out so the swatch re-roles with its lighting. */
const ROLES = [
	['background', 'bg-background'],
	['card', 'bg-card'],
	['muted', 'bg-muted'],
	['foreground', 'bg-foreground'],
	['muted-foreground', 'bg-muted-foreground'],
	['faint', 'bg-faint'],
	['primary', 'bg-primary'],
	['primary-soft', 'bg-primary-soft'],
	['signal', 'bg-signal'],
	['border', 'bg-border'],
] as const;

const TYPE = [
	['display', 'type-display', 'Hi! I’m Mathieu.'],
	['headline', 'type-headline', 'The Software Factory'],
	['title', 'type-title', 'Antifragility in the Age of Agentic Code'],
	['subtitle', 'type-subtitle', 'East of Eden'],
	[
		'reading',
		'type-reading',
		'At the turn of the 20th century, Frederick Taylor introduced Taylorism, a method for organizing factory work around maximizing productivity.',
	],
	['lead', 'type-lead', 'An engineer working in DeFi.'],
	['body', 'type-body', 'Senior Protocol Engineer at Zama'],
	['body-strong', 'type-body-strong', 'Kakarot'],
	['control', 'type-control', 'All writing'],
	['control-sm', 'type-control-sm', 'Copy'],
	['caption', 'type-caption', 'John Steinbeck'],
	['small', 'type-small', '412 pages'],
	['eyebrow', 'type-eyebrow', 'Last set'],
	['meta', 'type-meta', '11 Apr 2026 · 9 min read'],
	['figure', 'type-figure', '24,897'],
	['code', 'type-code', 'await program.aforward(query=query)'],
] as const;

const SPACE = [
	['inset', 'w-inset', 'Inside a card'],
	['gutter', 'w-gutter', 'Between cards'],
	['rhythm', 'w-rhythm', 'Between page blocks'],
] as const;

const RADII = [
	['card', 'rounded-card'],
	['media', 'rounded-media'],
	['row', 'rounded-row'],
	['chip', 'rounded-chip'],
	['control', 'rounded-control'],
] as const;

const DEPTH = [
	['card', 'shadow-card', 'Cards, pills, code'],
	['raised', 'shadow-raised', 'Primary buttons, dialogs'],
	['lifted', 'shadow-lifted', 'A card under the pointer'],
] as const;

/** The same content lit twice: by day and by night, whatever the page's own theme. */
function Lightings({ className, children }: { className?: string; children: ReactNode }) {
	return (
		<div className="grid gap-gutter sm:grid-cols-2">
			{(['light', 'dark'] as const).map((theme) => (
				<div
					key={theme}
					className={cn(theme, 'rounded-card bg-background p-inset text-foreground shadow-card')}
				>
					<Eyebrow as="p">{theme === 'light' ? 'Day' : 'Night'}</Eyebrow>
					<div className={cn('mt-4', className)}>{children}</div>
				</div>
			))}
		</div>
	);
}

function Note({ children }: { children: ReactNode }) {
	return <p className="max-w-prose type-body text-pretty text-muted-foreground">{children}</p>;
}

function Brand() {
	return (
		<div className="flex flex-col gap-gutter">
			<Lightings className="flex flex-col gap-6">
				<div className="flex items-end gap-4">
					<StickerMark className="size-16" />
					<StickerMark className="size-10" />
					<StickerMark className="size-7" />
					<StickerMark className="size-4" />
				</div>
				<div className="flex items-end gap-6">
					<Wordmark className="h-24" />
					<Wordmark />
				</div>
			</Lightings>
			<Note>
				The favicon is the wordmark’s M as a sticker of its own: the full wordmark is two lines, and
				at 16px it cannot be read. It keeps the sticker roles, the same in both themes.
			</Note>
			<Note>
				The header carries the sticker wordmark: Titan One, outlined so it needs no font, in the
				sticker roles, which stay the same in both themes.
			</Note>
		</div>
	);
}

function Typography() {
	return (
		<div className="flex flex-col gap-8">
			<div className="grid gap-gutter sm:grid-cols-3">
				{(
					[
						[TYPE_FAMILIES[0], 'type-display', 'Aa'],
						[TYPE_FAMILIES[1], 'type-reading', 'Aa'],
						[TYPE_FAMILIES[2], 'type-figure', '09'],
					] as const
				).map(([{ name, job: use }, role, glyph]) => (
					<div key={name} className="rounded-card bg-card p-inset shadow-card">
						<p aria-hidden className={cn(role, 'text-foreground')}>
							{glyph}
						</p>
						<p className="mt-6 type-body-strong text-foreground">{name}</p>
						<p className="type-caption text-faint">{use}</p>
					</div>
				))}
			</div>
			<Note>
				Components never set a size, weight or family. They name a role, and the role carries all of
				it. The specs below are read from the page.
			</Note>
			<dl className="flex flex-col">
				{TYPE.map(([role, className, sample]) => (
					<TypeSpec key={role} role={role} className={className} sample={sample} />
				))}
			</dl>
		</div>
	);
}

function Colour() {
	return (
		<div className="flex flex-col gap-8">
			<div className="flex flex-col gap-5">
				{PALETTES.map((palette) => (
					<div key={palette.name}>
						<div className="mb-2 flex items-baseline gap-2">
							<span className="type-body-strong text-foreground">{palette.name}</span>
							<span className="type-caption text-faint">{palette.job}</span>
						</div>
						<ol className="flex overflow-hidden rounded-media shadow-card">
							{palette.steps.map((step) => {
								const swatch = `var(--palette-${palette.name}-${step})`;
								return (
									<li
										key={step}
										title={`--palette-${palette.name}-${step}`}
										className="flex h-12 min-w-0 flex-1 items-end p-1 type-meta"
										style={{
											background: swatch,
											// Black ink on light steps, white on dark ones, from the step's own lightness.
											color: `oklch(from ${swatch} clamp(0, (0.62 - l) * 1000, 1) 0 0)`,
										}}
									>
										{step}
									</li>
								);
							})}
						</ol>
					</div>
				))}
			</div>
			<div className="flex flex-col gap-3">
				<Note>
					Palettes are named for their job, never their hue, and no component can name one. Each
					theme points a role at a step.
				</Note>
				<Lightings className="grid grid-cols-2 gap-x-3 gap-y-2">
					{ROLES.map(([name, bg]) => (
						<div key={name} className="flex items-center gap-2">
							<span className={cn(bg, 'size-5 flex-none rounded-chip shadow-card')} />
							<span className="truncate type-meta text-muted-foreground">{name}</span>
						</div>
					))}
				</Lightings>
			</div>
		</div>
	);
}

function Space() {
	return (
		<div className="flex flex-col gap-8">
			<ol className="flex flex-col gap-3">
				{SPACE.map(([name, width, use]) => (
					<li key={name} className="flex items-center gap-4">
						<span className="w-16 flex-none type-meta text-faint">{name}</span>
						<span className={cn(width, 'h-3 flex-none rounded-inline bg-primary')} />
						<span className="type-caption text-faint">{use}</span>
					</li>
				))}
			</ol>
			<div className="rounded-card bg-card p-inset shadow-card">
				<div className="grid h-32 place-items-center rounded-media bg-muted">
					<span className="type-meta text-faint">media = card − inset</span>
				</div>
				<p className="mt-3 type-body text-muted-foreground">
					Media inside a card takes the card's radius minus the inset, so the corners stay
					concentric.
				</p>
			</div>
			<ol className="flex flex-wrap gap-4">
				{RADII.map(([name, radius]) => (
					<li key={name} className="flex flex-col items-center gap-2">
						<span className={cn(radius, 'size-14 bg-card shadow-raised')} />
						<span className="type-meta text-faint">{name}</span>
					</li>
				))}
			</ol>
		</div>
	);
}

function Depth() {
	return (
		<div className="flex flex-col gap-3">
			<Lightings className="flex flex-col gap-3">
				{DEPTH.map(([name, shadow, use]) => (
					<div key={name} className={cn(shadow, 'rounded-row bg-card px-4 py-3')}>
						<p className="type-body-strong">{name}</p>
						<p className="type-caption text-faint">{use}</p>
					</div>
				))}
			</Lightings>
			<Note>
				By day, a ring and two shadows tinted by the neutral palette. Shadows vanish on a dark
				ground, so by night each level is a ring and a top highlight, and the lifted card glows.
			</Note>
		</div>
	);
}

function Components() {
	return (
		<Lightings className="flex flex-col gap-5">
			<div className="flex flex-wrap gap-2">
				<Button variant="primary">Primary</Button>
				<Button>Default</Button>
				<Button variant="ghost">Ghost</Button>
			</div>
			<div className="flex flex-wrap items-center gap-3">
				<Badge>Reading</Badge>
				<Badge>Latest</Badge>
				<span className="flex items-center gap-1.5 type-caption text-muted-foreground">
					<StatusDot live /> Live
				</span>
			</div>
			<Eyebrow dot meta="26 Sept 2026">
				Last set
			</Eyebrow>
		</Lightings>
	);
}

const SECTIONS = [
	['brand', 'Brand', 'Mark and wordmark', Brand],
	['type', 'Typography', 'Type roles over three families', Typography],
	['colour', 'Colour', 'Palettes and the roles that use them', Colour],
	['space', 'Space and shape', 'Inset, gutter, rhythm; five radii', Space],
	['depth', 'Depth', 'Three elevations, two lightings', Depth],
	['components', 'Components', 'Buttons, badges, eyebrows', Components],
] as const;

export default function DesignPage() {
	return (
		<Page>
			<PageHeader lead="The primitives this site is built from. Open a section to see it by day and by night.">
				Design
			</PageHeader>
			<Accordion multiple defaultValue={['brand']}>
				{SECTIONS.map(([value, title, summary, Section], i) => (
					<AccordionItem key={value} value={value}>
						<AccordionTrigger number={String(i + 1).padStart(2, '0')} summary={summary}>
							{title}
						</AccordionTrigger>
						<AccordionPanel>
							<Section />
						</AccordionPanel>
					</AccordionItem>
				))}
			</Accordion>
		</Page>
	);
}
