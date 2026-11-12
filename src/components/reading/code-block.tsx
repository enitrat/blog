'use client';

import { type ComponentProps, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';

/** Milliseconds "Copied!" stays before the button reads "Copy" again. */
const RESET = 2000;

const BUTTON = { idle: 'Copy', copied: 'Copied!', failed: 'Failed' } as const;
const STATUS = { idle: '', copied: 'Copied to clipboard', failed: 'Copy failed' } as const;

/**
 * A highlighted code block with a Copy button in its corner. The button appears once the page is
 * hydrated; without JS the code is still selectable. `data-copy-label` comes from `renderMarkdown`.
 */
export function CodeBlock({
	children,
	'data-copy-label': label = 'code sample',
	tabIndex: _shikiTabIndex,
	...props
}: ComponentProps<'pre'> & { 'data-copy-label'?: string }) {
	const ref = useRef<HTMLPreElement>(null);
	const [ready, setReady] = useState(false);
	const [overflows, setOverflows] = useState(false);
	const [state, setState] = useState<keyof typeof BUTTON>('idle');

	// A scroll box is reachable by keyboard only if it is focusable, but a block with nothing to
	// scroll would be a tab stop that does nothing.
	useEffect(() => {
		setReady(true);
		const el = ref.current;
		if (!el) return;
		const ro = new ResizeObserver(() => setOverflows(el.scrollWidth > el.clientWidth));
		ro.observe(el);
		return () => ro.disconnect();
	}, []);

	useEffect(() => {
		if (state === 'idle') return;
		const timer = setTimeout(() => setState('idle'), RESET);
		return () => clearTimeout(timer);
	}, [state]);

	const copy = async () => {
		try {
			await navigator.clipboard.writeText(ref.current?.textContent ?? '');
			setState('copied');
		} catch {
			setState('failed');
		}
	};

	return (
		<div className="relative">
			{/* biome-ignore lint/a11y/useSemanticElements: the block itself is the named region; a wrapping <section> would split it from its scroll box */}
			<pre
				ref={ref}
				{...props}
				role="region"
				aria-label={label}
				tabIndex={overflows ? 0 : undefined}
			>
				{children}
			</pre>
			{ready && (
				<>
					<Button
						variant="ghost"
						size="sm"
						className="absolute top-2 right-2 min-w-18 bg-card/80 backdrop-blur-sm after:absolute after:-inset-2 after:content-['']"
						aria-label={`Copy ${label}`}
						onClick={copy}
					>
						{BUTTON[state]}
					</Button>
					<span className="sr-only" role="status">
						{STATUS[state]}
					</span>
				</>
			)}
		</div>
	);
}
