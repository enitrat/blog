// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'astro/config';
import rehypeMermaid from 'rehype-mermaid';
import vercel from '@astrojs/vercel';

// https://astro.build/config
export default defineConfig({
  site: 'https://example.com',
  integrations: [mdx(), sitemap()],

  markdown: {
      rehypePlugins: [rehypeMermaid],
      syntaxHighlight: {
          type: 'shiki',
          excludeLangs: ['mermaid', 'math'],
      },
      shikiConfig: {
          themes: {
              light: 'catppuccin-latte',
              dark: 'github-dark',
          },
      },
	},

  vite: {
      plugins: [tailwindcss()],
	},
  adapter: vercel(),
});
