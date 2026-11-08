import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Prose } from '@/components/reading/prose';
import { Label, Page, PageTitle } from '@/ds';
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
	const { tree } = await renderMarkdown(
		readFileSync(join(process.cwd(), 'src/content/about.md'), 'utf8'),
		'.',
	);
	return (
		<Page width="measure" as="article">
			<PageTitle>About</PageTitle>
			<Prose tree={tree} />
			<dl className="m-0 grid gap-4 border-t border-border pt-6 text-ui">
				<div>
					<dt>
						<Label>Currently</Label>
					</dt>
					<dd className="mt-1">Senior Protocol Engineer at Zama</dd>
				</div>
			</dl>
		</Page>
	);
}
