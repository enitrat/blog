'use client';

import { useEffect, useRef } from 'react';
import { Button } from '@/ds';
import { initCopyCode } from './copy-code';

/**
 * A ghost sm `Button` on every code block of the article: the button is rendered once, hidden,
 * and the controller clones it into each block.
 */
export function CopyCodeButton() {
	const template = useRef<HTMLDivElement>(null);
	useEffect(() => {
		const article = template.current?.closest('article');
		return article && template.current ? initCopyCode(article, template.current) : undefined;
	}, []);
	return (
		<div ref={template} hidden>
			<Button
				variant="ghost"
				size="sm"
				// The block is mono; the button keeps the UI face.
				className="copy-btn absolute top-(--copy-code-inset) right-(--copy-code-inset) font-sans"
			>
				Copy
			</Button>
		</div>
	);
}
