import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';

/** Small non-interactive flag ("Reading", "Latest"). */
export function Badge({ className, ...props }: ComponentProps<'span'>) {
	return (
		<span
			className={cn(
				'inline-flex items-center whitespace-nowrap rounded-control bg-primary-soft px-2 py-0.5 type-control-sm text-primary',
				className,
			)}
			{...props}
		/>
	);
}
