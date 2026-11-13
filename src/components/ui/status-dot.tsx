import { cn } from '@/lib/utils';

/** Live-status dot; `live` adds the ping ring. */
export function StatusDot({ live = false, className }: { live?: boolean; className?: string }) {
	return (
		<span aria-hidden="true" className={cn('relative inline-flex size-1.5 flex-none', className)}>
			{live && (
				<span className="absolute inline-flex size-full animate-ping rounded-full bg-signal opacity-75 motion-only" />
			)}
			<span
				className={cn(
					'relative inline-flex size-1.5 rounded-full',
					live ? 'bg-signal' : 'bg-primary',
				)}
			/>
		</span>
	);
}
