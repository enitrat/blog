'use client';

import { MoonIcon, SunIcon } from 'lucide-react';
import { useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';

/**
 * Flips between light and dark. Landing back on the system's own scheme hands control back to the
 * system, so one toggle never pins a theme the visitor's OS would have picked anyway.
 */
export function ThemeToggle() {
	const { resolvedTheme, systemTheme, setTheme } = useTheme();
	const flip = () => {
		const next = resolvedTheme === 'dark' ? 'light' : 'dark';
		setTheme(next === systemTheme ? 'system' : next);
	};
	return (
		<Button variant="ghost" size="icon" aria-label="Toggle dark mode" onClick={flip}>
			<SunIcon className="dark:hidden" />
			<MoonIcon className="hidden dark:block" />
		</Button>
	);
}
