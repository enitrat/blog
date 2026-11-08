import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Prose } from '@/components/reading/prose';
import { Eyebrow } from '@/components/ui/eyebrow';
import { Page, PageHeader } from '@/components/ui/page';
import { renderMarkdown } from '@/lib/markdown';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'About',
	description: 'I build software, read books, and keep a few projects of my own alive.',
	path: '/about/',
});

// Mathieu's own prose, rendered unchanged. This page owns layout and typography only; the words
// and every claim in them stay as written.
export default async function AboutPage() {
	const tree = await renderMarkdown(
		readFileSync(join(process.cwd(), 'src/content/about.md'), 'utf8'),
		'.',
	);
	return (
		<Page width="text" as="article">
			<PageHeader>About</PageHeader>
			<Prose tree={tree} />
			<dl className="border-t border-gray-200 pt-6 text-sm dark:border-white/10">
				<dt>
					<Eyebrow>Currently</Eyebrow>
				</dt>
				<dd className="mt-1 text-gray-800 dark:text-iron-100">Senior Protocol Engineer at Zama</dd>
			</dl>
		</Page>
	);
}
