import { GeistMono } from 'geist/font/mono';
import { Figtree } from 'next/font/google';

export const sans = Figtree({ subsets: ['latin'], variable: '--font-figtree', display: 'swap' });

export const mono = GeistMono;
