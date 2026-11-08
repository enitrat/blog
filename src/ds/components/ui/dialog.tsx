'use client';

import { Dialog as DialogPrimitive } from '@base-ui/react/dialog';
import type { ComponentProps } from 'react';
import { cn } from '@/ds/lib/cn';

/** Keyboard, focus trap, dismiss and scroll lock come from Base UI. */
export function Dialog(props: DialogPrimitive.Root.Props) {
	return <DialogPrimitive.Root data-slot="dialog" {...props} />;
}

export function DialogTrigger(props: DialogPrimitive.Trigger.Props) {
	return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />;
}

export function DialogClose(props: DialogPrimitive.Close.Props) {
	return <DialogPrimitive.Close data-slot="dialog-close" {...props} />;
}

export function DialogOverlay({ className, ...props }: DialogPrimitive.Backdrop.Props) {
	return (
		<DialogPrimitive.Backdrop
			data-slot="dialog-overlay"
			className={cn(
				'fixed inset-0 z-(--z-dialog) bg-(--dialog-overlay) transition-opacity duration-base data-ending-style:opacity-0 data-starting-style:opacity-0',
				className,
			)}
			{...props}
		/>
	);
}

const SIZES = {
	/** A centred card. */
	md: 'top-1/2 left-1/2 w-[min(100%-2*var(--dialog-inset),var(--dialog-max-w))] -translate-1/2 rounded-(--dialog-radius) border border-(--dialog-edge) p-(--dialog-pad)',
	/** Fills the viewport less the page gutter: a photo, the room. */
	full: 'inset-(--dialog-inset) rounded-(--dialog-radius)',
} as const;

export function DialogContent({
	className,
	children,
	size = 'md',
	closeLabel = 'Close',
	showCloseButton = true,
	...props
}: DialogPrimitive.Popup.Props & {
	size?: keyof typeof SIZES;
	closeLabel?: string;
	showCloseButton?: boolean;
}) {
	return (
		<DialogPrimitive.Portal>
			<DialogOverlay />
			<DialogPrimitive.Popup
				data-slot="dialog-content"
				className={cn(
					'fixed z-(--z-dialog) grid gap-(--dialog-gap) bg-(--dialog-bg) text-(--dialog-fg) outline-none',
					'transition-[opacity,scale] duration-base data-ending-style:opacity-0 data-starting-style:opacity-0 motion-safe:data-ending-style:scale-(--dialog-pop-scale) motion-safe:data-starting-style:scale-(--dialog-pop-scale)',
					SIZES[size],
					className,
				)}
				{...props}
			>
				{children}
				{showCloseButton && (
					<DialogPrimitive.Close
						data-slot="dialog-close"
						className="press focus-ring hit-area absolute top-3 right-3 inline-flex size-(--control-h-sm) cursor-pointer items-center justify-center rounded-pill bg-transparent text-foreground hovered:bg-highlight"
					>
						<svg
							aria-hidden="true"
							viewBox="0 0 24 24"
							className="size-(--icon-md) fill-none stroke-current stroke-(length:--icon-stroke) [stroke-linecap:round]"
						>
							<path d="M6 6l12 12M18 6L6 18" />
						</svg>
						<span className="sr-only">{closeLabel}</span>
					</DialogPrimitive.Close>
				)}
			</DialogPrimitive.Popup>
		</DialogPrimitive.Portal>
	);
}

export function DialogHeader({ className, ...props }: ComponentProps<'div'>) {
	return <div data-slot="dialog-header" className={cn('grid gap-2 pe-10', className)} {...props} />;
}

export function DialogTitle({ className, ...props }: DialogPrimitive.Title.Props) {
	return (
		<DialogPrimitive.Title
			data-slot="dialog-title"
			className={cn('type-heading', className)}
			{...props}
		/>
	);
}

export function DialogDescription({ className, ...props }: DialogPrimitive.Description.Props) {
	return (
		<DialogPrimitive.Description
			data-slot="dialog-description"
			className={cn('type-small text-muted-foreground', className)}
			{...props}
		/>
	);
}
