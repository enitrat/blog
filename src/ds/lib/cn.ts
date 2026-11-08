import type { CnFunction } from 'cn';
import { createCn } from 'cn/config';

/**
 * Knows the ds scales in `tokens/`, so a role size beside a colour (`text-label text-foreground`)
 * or `shadow-none` over `shadow-tile` merge correctly. Every ds and site file imports `cn` from here.
 */
export const cn: CnFunction = createCn({
	extend: {
		theme: {
			text: [
				'xs',
				'sm',
				'base',
				'md',
				'lg',
				'xl',
				'2xl',
				'3xl',
				'display',
				'title',
				'heading',
				'body',
				'ui',
				'small',
				'label',
			],
		},
		classGroups: {
			shadow: [{ shadow: ['tile', 'tile-lift', 'object', 'object-lift'] }],
			ease: [{ ease: ['spring'] }],
			duration: [{ duration: ['fast', 'base', 'moderate', 'slow'] }],
			type: [
				{
					type: ['display', 'title', 'heading', 'body', 'ui', 'control', 'small', 'label', 'data'],
				},
			],
		},
	},
});
