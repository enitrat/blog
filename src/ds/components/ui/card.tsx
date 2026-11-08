import { mergeProps } from '@base-ui/react/merge-props';
import { useRender } from '@base-ui/react/use-render';
import { cn } from '@/ds/lib/cn';

const VARIANTS = {
	/** A surface with a hairline. */
	default: '',
	/** An object on a tile: casts `--elevation-object`. */
	object: 'shadow-object',
	/** A whole-card target; pass `render` with a link or a button. */
	interactive:
		'focus-ring cursor-pointer transition-[border-color,box-shadow,transform] duration-base hovered:border-(--card-edge-hover) hovered:shadow-object-lift motion-safe:active:scale-(--press-scale)',
} as const;

export type CardVariant = keyof typeof VARIANTS;

export type CardProps = useRender.ComponentProps<'div'> & { variant?: CardVariant };

/** A bounded surface. A bench surface with a label header is a `Tile`; a card is smaller. */
export function Card({ variant = 'default', className, render, ...props }: CardProps) {
	return useRender({
		defaultTagName: 'div',
		render,
		props: mergeProps<'div'>(
			{
				className: cn(
					'flex min-w-0 flex-col gap-(--card-gap) rounded-(--card-radius) border border-(--card-edge) bg-(--card-bg) p-(--card-pad) text-(--card-fg)',
					VARIANTS[variant],
					className,
				),
			},
			props,
		),
		state: { slot: 'card' },
	});
}

type PartProps = useRender.ComponentProps<'div'>;

function part(slot: string, recipe: string, { className, render, ...props }: PartProps) {
	return { slot, recipe, className, render, props };
}

function usePart(p: ReturnType<typeof part>) {
	return useRender({
		defaultTagName: 'div',
		render: p.render,
		props: mergeProps<'div'>({ className: cn(p.recipe, p.className) }, p.props),
		state: { slot: p.slot },
	});
}

export function CardHeader(props: PartProps) {
	return usePart(part('card-header', 'grid gap-1', props));
}

export function CardTitle(props: PartProps) {
	return usePart(part('card-title', 'text-ui font-semibold leading-snug tracking-snug', props));
}

export function CardDescription(props: PartProps) {
	return usePart(part('card-description', 'type-small text-(--card-description-fg)', props));
}

export function CardContent(props: PartProps) {
	return usePart(part('card-content', 'min-w-0', props));
}

export function CardFooter(props: PartProps) {
	return usePart(part('card-footer', 'flex items-center gap-2', props));
}
