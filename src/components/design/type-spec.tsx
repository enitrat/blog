'use client';

import { useEffect, useRef, useState } from 'react';
import { TYPE_FAMILIES } from '@/lib/type-families';
import { cn } from '@/lib/utils';

/** Reads what the browser resolved for a type role, so the spec can never drift from the token. */
function describe(el: HTMLElement) {
	const s = getComputedStyle(el);
	const family = TYPE_FAMILIES.find((f) => s.fontFamily.includes(f.name))?.name ?? 'System';
	const size = Number.parseFloat(s.fontSize);
	const leading = Number.parseFloat(s.lineHeight);
	const tracking = Number.parseFloat(s.letterSpacing);
	const parts = [
		`${family} ${s.fontWeight}`,
		`${Math.round(size)}/${Math.round(leading)}`,
		Number.isFinite(tracking) && tracking !== 0
			? `${tracking > 0 ? '+' : '−'}${Math.abs((tracking / size) * 100)
					.toFixed(1)
					.replace(/\.0$/, '')}%`
			: null,
		s.textTransform === 'uppercase' ? 'caps' : null,
		s.fontVariantNumeric.includes('tabular') ? 'tabular' : null,
		s.fontVariantNumeric.includes('oldstyle') ? 'oldstyle' : null,
	];
	return parts.filter(Boolean).join(' · ');
}

/** A type role: its name, its resolved spec, and a sample set in it. */
export function TypeSpec({
	role,
	className,
	sample,
}: {
	role: string;
	/** The role's utility, written out in full so Tailwind sees it. */
	className: string;
	sample: string;
}) {
	const ref = useRef<HTMLElement>(null);
	const [spec, setSpec] = useState('');
	useEffect(() => {
		const el = ref.current;
		if (!el) return;
		const update = () => setSpec(describe(el));
		update();
		document.fonts.ready.then(update);
		const observer = new ResizeObserver(update);
		observer.observe(el);
		return () => observer.disconnect();
	}, []);
	return (
		<div className="flex flex-col gap-2 border-t border-border py-4 sm:flex-row sm:items-baseline sm:gap-6">
			<dt className="flex flex-col sm:w-52 sm:flex-none">
				<span className="type-body-strong text-foreground">{role}</span>
				<span className="type-meta text-faint">{spec || ' '}</span>
			</dt>
			<dd ref={ref} className={cn(className, 'min-w-0 text-balance text-foreground')}>
				{sample}
			</dd>
		</div>
	);
}
