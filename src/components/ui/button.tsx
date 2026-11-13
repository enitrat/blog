import { Button as ButtonPrimitive } from '@base-ui/react/button';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

export const buttonVariants = cva(
	'inline-flex shrink-0 cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap rounded-control type-control transition-[color,background-color,box-shadow,scale] duration-150 ease-out active:scale-[0.96] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0',
	{
		variants: {
			variant: {
				/** Card pill. */
				default: 'bg-card text-foreground shadow-card hover:bg-muted',
				/** Solid accent. */
				primary: 'bg-primary text-primary-foreground shadow-raised hover:bg-primary/90',
				/** See-through card pill, for over images and sticky chrome. */
				glass:
					'bg-card/60 text-foreground shadow-card backdrop-blur-xs backdrop-saturate-150 hover:bg-card/80',
				ghost: 'text-muted-foreground hover:bg-muted hover:text-foreground',
			},
			size: {
				sm: 'h-7 px-3 type-control-sm [&_svg]:size-3.5',
				default: 'h-9 pt-2 pr-4 pb-1.5 pl-3.5 [&_svg]:size-4',
				/** 36px drawn, 44px to press. */
				icon: 'relative size-9 hit-area [&_svg]:size-4',
				/** 28px drawn, 44px to press. */
				'icon-sm': 'relative size-7 hit-area [&_svg]:size-3.5',
			},
		},
		defaultVariants: { variant: 'default', size: 'default' },
	},
);

export function Button({
	className,
	variant,
	size,
	...props
}: ButtonPrimitive.Props & VariantProps<typeof buttonVariants>) {
	return (
		<ButtonPrimitive
			data-slot="button"
			className={cn(buttonVariants({ variant, size }), className)}
			{...props}
		/>
	);
}
