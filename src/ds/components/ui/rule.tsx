import { Separator as SeparatorPrimitive } from '@base-ui/react/separator';
import { cn } from '@/ds/lib/cn';

/** Horizontal hairline in `--border`; retints on any inverse surface. */
export function Rule({ className, ...props }: SeparatorPrimitive.Props) {
	return (
		<SeparatorPrimitive
			data-slot="rule"
			className={cn('h-0 border-0 border-t border-border', className)}
			{...props}
		/>
	);
}
