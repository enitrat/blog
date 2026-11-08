'use client';

import { useEffect, useRef } from 'react';
import { Icon, Link } from '@/ds';
import { cn } from '@/ds/lib/cn';
import { IN_SHORT_LABEL } from '@/utils/inShort';

/** Adds a quiet "In short" back link at the end of each section; absent without JS. */
function addBackLinks(prose: HTMLElement, template: HTMLElement) {
	if (!document.getElementById('in-short')) return () => {};
	const added: Element[] = [];
	const h2s = [...prose.querySelectorAll(':scope > h2')];
	for (const h2 of [...h2s.slice(1), null]) {
		const up = template.firstElementChild?.cloneNode(true) as Element;
		h2 ? h2.before(up) : prose.append(up);
		added.push(up);
	}
	return () => {
		for (const el of added) el.remove();
	};
}

/**
 * Essay typography for rendered markdown: Geist at the body role, one `--measure` column,
 * left-aligned and unhyphenated. `html` is trusted build-time output of `renderMarkdown`.
 */
export function Prose({ html, className }: { html: string; className?: string }) {
	const ref = useRef<HTMLDivElement>(null);
	const up = useRef<HTMLDivElement>(null);
	useEffect(
		() => (ref.current && up.current ? addBackLinks(ref.current, up.current) : undefined),
		[],
	);
	return (
		<>
			<div
				ref={ref}
				className={cn('prose', className)}
				// biome-ignore lint/security/noDangerouslySetInnerHtml: build-time markdown from src/content
				dangerouslySetInnerHTML={{ __html: html }}
			/>
			<div ref={up} hidden>
				<p className="up">
					<Link href="#in-short" variant="quiet">
						<Icon name="arrow-up" size="sm" /> {IN_SHORT_LABEL}
					</Link>
				</p>
			</div>
		</>
	);
}
