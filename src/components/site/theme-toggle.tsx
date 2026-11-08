'use client';

import { MoonIcon, SunIcon } from 'lucide-react';
import { useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';

export function ThemeToggle() {
	const { resolvedTheme, setTheme } = useTheme();
	return (
		<Button
			variant="glass"
			size="icon"
			aria-label="Toggle dark mode"
			// A 44px hit area around the 36px button.
			className="relative after:absolute after:-inset-1 after:content-['']"
			onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
		>
			<SunIcon className="dark:hidden" />
			<MoonIcon className="hidden dark:block" />
		</Button>
	);
}
