'use client';

import { MotionConfig } from 'motion/react';
import { ThemeProvider } from 'next-themes';
import type { ReactNode } from 'react';

/**
 * Site-wide behaviour that every component inherits instead of re-deciding it.
 * - Theme: the system's by default; the toggle overrides it and hands it back (see ThemeToggle).
 * - Reduced motion: motion components skip transform and layout animation for users who ask, the
 *   same rule globals.css applies to CSS transitions. No component checks it for animation.
 */
export function Providers({ children }: { children: ReactNode }) {
	return (
		<ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
			<MotionConfig reducedMotion="user">{children}</MotionConfig>
		</ThemeProvider>
	);
}
