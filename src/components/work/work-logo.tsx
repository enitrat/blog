import 'server-only';
import Image from 'next/image';
import { maybeImage } from '@/lib/media';
import { cn } from '@/lib/utils';

const SIZES = {
	row: 'size-11 rounded-media text-lg',
	header: 'size-16 rounded-2xl text-2xl',
} as const;

/** A company's logo in a rounded square, or its first letter when no logo is checked in. */
export async function WorkLogo({
	id,
	title,
	size = 'row',
}: {
	id: string;
	title: string;
	size?: keyof typeof SIZES;
}) {
	const logo = await maybeImage(`logos/${id}.png`);
	const box = cn('flex-none overflow-hidden border border-border', SIZES[size]);
	if (logo) {
		return (
			<Image
				src={logo}
				alt=""
				sizes={size === 'row' ? '44px' : '64px'}
				className={cn(box, 'bg-card object-cover')}
			/>
		);
	}
	return (
		<span
			aria-hidden
			className={cn(
				box,
				'flex items-center justify-center bg-muted font-display text-muted-foreground',
			)}
		>
			{title.charAt(0).toUpperCase()}
		</span>
	);
}
