import { Button as ButtonPrimitive } from '@base-ui/react/button';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

export const buttonVariants = cva(
	'inline-flex shrink-0 cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap rounded-full font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-50 disabled:pointer-events-none disabled:opacity-50 dark:focus-visible:ring-offset-iron-950 [&_svg]:pointer-events-none [&_svg]:shrink-0',
	{
		variants: {
			variant: {
				/** White pill with a hairline and the signature shadow-xs. */
				default:
					'border border-gray-200 bg-white text-gray-800 shadow-xs hover:bg-gray-100/50 dark:border-white/10 dark:bg-iron-900 dark:text-iron-100 dark:hover:bg-iron-800',
				/** Brand gradient tile. */
				brand:
					'bg-linear-to-b from-brand-400 to-brand-500 text-white shadow-md hover:from-brand-500 hover:to-brand-600',
				/** Light glass pill. */
				glass:
					'border border-gray-200/50 bg-white/50 text-gray-800 backdrop-blur-xs backdrop-saturate-150 hover:bg-white/70 dark:border-iron-600/25 dark:bg-iron-950/65 dark:text-iron-100',
				ghost:
					'text-gray-600 hover:bg-gray-100 hover:text-gray-800 dark:text-iron-300 dark:hover:bg-iron-800 dark:hover:text-iron-100',
			},
			size: {
				sm: 'h-7 px-3 text-xs [&_svg]:size-3.5',
				default: 'h-9 pt-2 pr-4 pb-1.5 pl-3.5 text-sm [&_svg]:size-4',
				icon: 'size-9 [&_svg]:size-4',
				'icon-sm': 'size-7 [&_svg]:size-3.5',
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
