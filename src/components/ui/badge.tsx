import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';

/** Small non-interactive flag ("Reading", "Latest"). */
export function Badge({ className, ...props }: ComponentProps<'span'>) {
	return (
		<span
			className={cn(
				'inline-flex items-center whitespace-nowrap rounded-full bg-brand-500/10 px-2 py-0.5 text-xs font-medium text-brand-600 dark:bg-brand-400/15 dark:text-brand-400',
				className,
			)}
			{...props}
		/>
	);
}
