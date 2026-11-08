'use client';

import { Button as ButtonPrimitive } from '@base-ui/react/button';
import { cn } from '@/ds/lib/cn';

const VARIANTS = {
	primary: cn(
		'bg-(--button-primary-bg) text-(--button-primary-fg) hovered:bg-(--button-primary-bg-hover)',
		'aria-pressed:bg-(--button-primary-bg-pressed) aria-pressed:text-(--button-primary-fg-pressed)',
	),
	ghost: cn(
		'edge bg-(--button-ghost-bg) text-(--button-ghost-fg) [--edge:var(--button-ghost-edge)] hovered:[--edge:var(--button-ghost-edge-hover)]',
		'aria-pressed:bg-(--button-ghost-bg-pressed) aria-pressed:text-(--button-ghost-fg-pressed) aria-pressed:[--edge:transparent]',
	),
	inverse: cn(
		'bg-(--button-inverse-bg) text-(--button-inverse-fg) hovered:bg-(--button-inverse-bg-hover) hovered:text-(--button-inverse-fg-hover)',
		'aria-pressed:bg-(--button-inverse-bg-hover) aria-pressed:text-(--button-inverse-fg-hover)',
	),
} as const;

const SIZES = {
	md: 'min-h-(--button-h-md) px-(--button-px-md) text-ui',
	sm: 'hit-area min-h-(--button-h-sm) px-(--button-px-sm) text-small',
} as const;

export type ButtonVariant = keyof typeof VARIANTS;
export type ButtonSize = keyof typeof SIZES;

export type ButtonProps = Omit<ButtonPrimitive.Props, 'render'> & {
	variant?: ButtonVariant;
	size?: ButtonSize;
	/** Renders an `<a>`. A disabled link drops its href and reports `aria-disabled`. */
	href?: string;
	/** Toggle state, as `aria-pressed`. */
	pressed?: boolean;
};

/** Pill action. md is 44px tall; sm keeps a 44px hit area. */
export function Button({
	variant = 'primary',
	size = 'md',
	href,
	pressed,
	disabled,
	className,
	...props
}: ButtonProps) {
	const classes = cn(
		'press focus-ring relative inline-flex cursor-pointer items-center justify-center gap-(--button-gap) whitespace-nowrap rounded-(--button-radius) border-0 font-medium leading-none no-underline',
		'disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-(--opacity-disabled) aria-disabled:pointer-events-none aria-disabled:opacity-(--opacity-disabled)',
		SIZES[size],
		VARIANTS[variant],
		className,
	);
	if (href !== undefined) {
		return (
			<ButtonPrimitive
				data-slot="button"
				data-variant={variant}
				nativeButton={false}
				render={<a href={disabled ? undefined : href} aria-disabled={disabled || undefined} />}
				role={disabled ? 'link' : undefined}
				className={classes}
				{...props}
			/>
		);
	}
	return (
		<ButtonPrimitive
			data-slot="button"
			data-variant={variant}
			type="button"
			disabled={disabled}
			aria-pressed={pressed}
			className={classes}
			{...props}
		/>
	);
}
