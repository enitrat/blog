import { type ClassValue, clsx } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

// The @theme text sizes, so `text-eyebrow` is not mistaken for a colour and dropped beside one.
const twMerge = extendTailwindMerge({
	extend: { theme: { text: ['display', 'feature', 'meta', 'eyebrow'] } },
});

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}
