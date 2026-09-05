import * as THREE from 'three';
import { type Book, books } from '../../../booksData';
import { type PleiadeColor, pleiadeStyleFor } from '../../../utils/pleiade';

const colors: Record<PleiadeColor, string> = {
	green: '#416653',
	violet: '#574363',
	corinthe: '#70483f',
	red: '#813b34',
	blue: '#315a78',
	emerald: '#2f6652',
	havane: '#76533e',
	grey: '#62615d',
};

type Volume = {
	book: Book;
	group: THREE.Group;
	home: THREE.Vector3;
	button: HTMLButtonElement;
	width: number;
	height: number;
	color: string;
	cover: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
};

type Motion = {
	volume: Volume;
	from: THREE.Vector3;
	to: THREE.Vector3;
	rotationFrom: THREE.Quaternion;
	rotationTo: THREE.Quaternion;
	scaleFrom: number;
	scaleTo: number;
	start: number;
	duration: number;
	done: () => void;
};

function texture(width: number, height: number, draw: (ctx: CanvasRenderingContext2D) => void) {
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('Canvas 2D is required to draw book textures.');
	draw(ctx);
	const result = new THREE.CanvasTexture(canvas);
	result.colorSpace = THREE.SRGBColorSpace;
	return result;
}

function lettering(
	ctx: CanvasRenderingContext2D,
	text: string,
	x: number,
	y: number,
	width: number,
	line: number,
) {
	const words = text.split(/\s+/);
	let current = '';
	for (const word of words) {
		const next = current ? `${current} ${word}` : word;
		if (ctx.measureText(next).width > width && current) {
			ctx.fillText(current, x, y, width);
			y += line;
			current = word;
		} else current = next;
	}
	ctx.fillText(current, x, y, width);
}

function coverTexture(book: Book, color: string) {
	return texture(512, 768, (ctx) => {
		ctx.fillStyle = color;
		ctx.fillRect(0, 0, 512, 768);
		ctx.strokeStyle = '#baa46b';
		ctx.lineWidth = 2;
		ctx.strokeRect(18, 18, 476, 732);
		ctx.fillStyle = '#e6cf8e';
		ctx.textAlign = 'center';
		ctx.font = '20px "Literata Variable", Georgia';
		lettering(ctx, book.author.toUpperCase(), 256, 250, 410, 30);
		ctx.font = '42px "Literata Variable", Georgia';
		lettering(ctx, book.title, 256, 340, 405, 54);
		ctx.font = '20px "Literata Variable", Georgia';
		ctx.fillText('Bibliothèque personnelle', 256, 575);
	});
}

/** Mount the disposable library study. The scene renders only while a book moves. */
export async function startLibrary() {
	const stage = document.getElementById('library-stage');
	const canvas = document.getElementById('library-canvas');
	const targets = document.getElementById('book-targets');
	const caption = document.getElementById('book-caption');
	const loading = document.getElementById('library-loading');
	const dialog = document.getElementById('book-reader');
	const mount = document.getElementById('reader-mount');
	const title = document.getElementById('reader-title');
	const pageStatus = document.getElementById('page-status');
	const close = document.getElementById('return-book');
	const previous = document.getElementById('previous-page');
	const next = document.getElementById('next-page');
	if (
		!(canvas instanceof HTMLCanvasElement) ||
		!(dialog instanceof HTMLDialogElement) ||
		!stage ||
		!targets ||
		!caption ||
		!loading ||
		!mount ||
		!title ||
		!pageStatus ||
		!(close instanceof HTMLButtonElement) ||
		!(previous instanceof HTMLButtonElement) ||
		!(next instanceof HTMLButtonElement)
	)
		return;

	let renderer: THREE.WebGLRenderer;
	try {
		renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
	} catch {
		loading.textContent = 'The 3D view is unavailable. The reading archive is linked below.';
		return;
	}
	renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.1;
	await document.fonts.ready;

	const reduced = matchMedia('(prefers-reduced-motion: reduce)');
	const hover = matchMedia('(hover: hover) and (pointer: fine)');
	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera(-10, 10, 8, -8, 0.1, 100);
	scene.add(new THREE.HemisphereLight('#fff0ce', '#493b2c', 2.1));
	const sun = new THREE.DirectionalLight('#ffe3af', 2.8);
	sun.position.set(-8, 14, 12);
	scene.add(sun);
	const fill = new THREE.DirectionalLight('#eee8dd', 0.7);
	fill.position.set(8, 5, 3);
	scene.add(fill);

	const oakTexture = texture(512, 512, (ctx) => {
		ctx.fillStyle = '#65482f';
		ctx.fillRect(0, 0, 512, 512);
		for (let i = 0; i < 900; i++) {
			const seed = Math.sin(i * 127.1) * 43758.5453;
			const noise = seed - Math.floor(seed);
			ctx.strokeStyle =
				i % 3 ? `rgba(34,18,6,${0.03 + noise * 0.13})` : `rgba(230,177,103,${noise * 0.2})`;
			ctx.lineWidth = 0.4 + noise * 1.6;
			ctx.beginPath();
			for (let x = 0; x <= 512; x += 8) {
				const y = noise * 512 + Math.sin(x / 88 + i * 0.6) * 2.8 + Math.sin(x / 220 + i) * 5;
				if (x === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.stroke();
		}
	});
	const oak = new THREE.MeshStandardMaterial({ map: oakTexture, roughness: 0.8 });
	const verticalGrain = oakTexture.clone();
	verticalGrain.center.set(0.5, 0.5);
	verticalGrain.rotation = Math.PI / 2;
	const verticalOak = new THREE.MeshStandardMaterial({ map: verticalGrain, roughness: 0.8 });
	const darkOak = new THREE.MeshStandardMaterial({ color: '#4c3421', roughness: 1 });
	const trim = new THREE.MeshStandardMaterial({ color: '#9b6d3d', roughness: 0.65 });
	const backTexture = texture(64, 256, (ctx) => {
		const gradient = ctx.createLinearGradient(0, 0, 0, 256);
		gradient.addColorStop(0, '#231a12');
		gradient.addColorStop(0.65, '#624329');
		gradient.addColorStop(1, '#38261a');
		ctx.fillStyle = gradient;
		ctx.fillRect(0, 0, 64, 256);
	});
	const backMaterial = new THREE.MeshBasicMaterial({ map: backTexture });
	const caseGroup = new THREE.Group();
	scene.add(caseGroup);
	const volumes: Volume[] = [];
	let selected: Volume | undefined;
	let hovered: Volume | undefined;
	let motion: Motion | undefined;
	let frame = 0;
	let drawShelf = true;
	let iframe: HTMLIFrameElement | undefined;
	let flight:
		| { renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.OrthographicCamera }
		| undefined;
	let readerTimeout = 0;
	let generation = 0;
	let mobile = stage.clientWidth < 600;
	let viewHeight = 12;

	const box = (
		width: number,
		height: number,
		depth: number,
		x: number,
		y: number,
		z: number,
		material: THREE.Material,
	) => {
		const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
		mesh.position.set(x, y, z);
		caseGroup.add(mesh);
		return mesh;
	};

	for (const [index, book] of books.entries()) {
		const style = pleiadeStyleFor(book.author);
		const color = colors[style.color];
		const width = 0.47 + Math.min(book.edition.pageCount, 1200) / 6000;
		const height = 1.85 + (index % 3) * 0.06;
		const group = new THREE.Group();
		const body = new THREE.Mesh(
			new THREE.BoxGeometry(width, height, 1.2),
			new THREE.MeshStandardMaterial({ color, roughness: 0.85 }),
		);
		group.add(body);
		const spineTexture = texture(128, 512, (ctx) => {
			ctx.fillStyle = color;
			ctx.fillRect(0, 0, 128, 512);
			const shade = ctx.createLinearGradient(0, 0, 128, 0);
			shade.addColorStop(0, '#00000066');
			shade.addColorStop(0.3, '#ffffff10');
			shade.addColorStop(0.65, '#ffffff00');
			shade.addColorStop(1, '#00000066');
			ctx.fillStyle = shade;
			ctx.fillRect(0, 0, 128, 512);
			ctx.strokeStyle = '#ddc78b';
			ctx.lineWidth = 1;
			for (const y of [24, 32, 98, 105, 386, 393, 477, 485]) {
				ctx.beginPath();
				ctx.moveTo(7, y);
				ctx.lineTo(121, y);
				ctx.stroke();
			}
			ctx.strokeRect(7, 10, 114, 492);
			ctx.fillStyle = '#f0dfa5';
			ctx.textAlign = 'center';
			ctx.font = '18px "Literata Variable", Georgia';
			lettering(ctx, style.label.toUpperCase(), 64, 150, 104, 25);
			ctx.font = '17px "Literata Variable", Georgia';
			lettering(ctx, book.title.replace(/\s*\(.*/, ''), 64, 250, 103, 23);
			ctx.font = '12px Georgia';
			ctx.fillText('PLÉIADE', 64, 438);
		});
		const spine = new THREE.Mesh(
			new THREE.PlaneGeometry(width, height),
			new THREE.MeshBasicMaterial({ map: spineTexture }),
		);
		spine.position.z = 0.605;
		group.add(spine);

		const cover = new THREE.Mesh(
			new THREE.PlaneGeometry(1.2, height),
			new THREE.MeshBasicMaterial({ color }),
		);
		cover.rotation.y = Math.PI / 2;
		cover.position.x = width / 2 + 0.005;
		group.add(cover);
		const button = document.createElement('button');
		button.type = 'button';
		button.className = 'book-target';
		button.setAttribute('aria-label', `Open ${book.title}, ${book.author}`);
		button.setAttribute('aria-haspopup', 'dialog');
		targets.append(button);
		const volume: Volume = {
			book,
			group,
			home: new THREE.Vector3(),
			button,
			width,
			height,
			color,
			cover,
		};
		volumes.push(volume);
		scene.add(group);
		button.addEventListener('pointerenter', () => {
			if (hover.matches) pointAt(volume);
		});
		button.addEventListener('pointerleave', () => {
			if (hovered === volume) pointAt(undefined);
		});
		button.addEventListener('focus', () => pointAt(volume));
		button.addEventListener('blur', () => {
			if (hovered === volume) pointAt(undefined);
		});
		button.addEventListener('click', () => openBook(volume));
	}

	const pointAt = (volume: Volume | undefined) => {
		if (selected) return;
		hovered = volume;
		caption.textContent = volume
			? `${volume.book.title} · ${volume.book.author}`
			: 'Hover to explore · Select to open';
		requestRender();
	};

	const layout = () => {
		drawShelf = true;
		mobile = stage.clientWidth < 600;
		const columns = mobile ? 5 : 14;
		const rows = Math.ceil(volumes.length / columns);
		const width = mobile ? 4.55 : 12.15;
		const height = rows * 2.4 + 0.55;
		stage.style.height = mobile
			? `${Math.round(((stage.clientWidth * height) / width) * 0.95)}px`
			: '';
		for (const child of [...caseGroup.children]) {
			if (child instanceof THREE.Mesh) child.geometry.dispose();
			caseGroup.remove(child);
		}
		box(width, height, 0.14, 0, height / 2, -0.8, darkOak);
		for (let row = 0; row < rows; row++) {
			box(width - 0.3, 2.2, 0.03, 0, row * 2.4 + 1.28, -0.7, backMaterial);
		}
		for (let row = 0; row <= rows; row++) {
			box(width + 0.1, 0.2, 1.85, 0, row * 2.4 + 0.25, 0.05, oak);
			box(width + 0.16, 0.055, 0.07, 0, row * 2.4 + 0.32, 1.005, trim);
			box(width, 0.08, 0.1, 0, row * 2.4 + 0.15, 0.98, darkOak);
		}
		for (const x of [-width / 2, width / 2, ...(mobile ? [] : [0])]) {
			box(0.22, height, 1.9, x, height / 2, 0.05, verticalOak);
			box(0.08, height - 0.1, 0.055, x, height / 2, 1.02, trim);
		}
		box(width + 0.45, 0.18, 2.04, 0, height + 0.025, 0.03, oak);
		box(width + 0.3, 0.1, 1.98, 0, height - 0.13, 0.03, trim);
		box(width + 0.3, 0.3, 2, 0, 0.01, 0.03, oak);
		for (const [i, volume] of volumes.entries()) {
			const column = i % columns;
			const x = mobile ? -1.6 + column * 0.79 : -5.48 + column * 0.79 + (column >= 7 ? 0.65 : 0);
			volume.home.set(
				x,
				(rows - 1 - Math.floor(i / columns)) * 2.4 + 0.36 + volume.height / 2,
				0.16,
			);
			volume.group.position.copy(volume.home);
			volume.group.quaternion.identity();
			volume.group.scale.setScalar(1);
		}
		const aspect = stage.clientWidth / stage.clientHeight;
		viewHeight = Math.max(
			height * (mobile ? 1.02 : 1.09),
			(width + (mobile ? 0.15 : 1.4)) / aspect,
		);
		camera.left = (-viewHeight * aspect) / 2;
		camera.right = (viewHeight * aspect) / 2;
		camera.top = viewHeight / 2;
		camera.bottom = -viewHeight / 2;
		camera.position.set(mobile ? 3.2 : 5.5, height / 2 + 3.1, 26);
		camera.lookAt(0, height / 2, 0);
		camera.updateProjectionMatrix();
		camera.updateMatrixWorld();
		renderer.setSize(stage.clientWidth, stage.clientHeight, false);
		for (const volume of volumes) {
			const center = volume.home
				.clone()
				.add(new THREE.Vector3(0, 0, 0.62))
				.project(camera);
			const w = (volume.width / (camera.right - camera.left)) * stage.clientWidth;
			const h = (volume.height / viewHeight) * stage.clientHeight;
			Object.assign(volume.button.style, {
				left: `${((center.x + 1) / 2) * stage.clientWidth - w / 2}px`,
				top: `${((1 - center.y) / 2) * stage.clientHeight - h / 2}px`,
				width: `${Math.max(w, 22)}px`,
				height: `${h}px`,
			});
		}
		requestRender();
	};

	const requestRender = () => {
		if (!frame) frame = requestAnimationFrame(render);
	};
	const render = (now: number) => {
		frame = 0;
		let moving = false;
		for (const volume of volumes) {
			if (volume === selected) continue;
			const target = volume.home.z + (volume === hovered && !reduced.matches ? 0.5 : 0);
			const difference = target - volume.group.position.z;
			volume.group.position.z =
				Math.abs(difference) < 0.002 || reduced.matches
					? target
					: volume.group.position.z + difference * 0.24;
			if (difference !== 0) drawShelf = true;
			if (Math.abs(difference) >= 0.002) moving = true;
		}
		if (motion) {
			const current = motion;
			const t = Math.min(1, (now - current.start) / current.duration);
			// Cubic ease-out gives immediate extraction feedback without a spring overshoot.
			const eased = 1 - (1 - t) ** 3;
			current.volume.group.position.lerpVectors(current.from, current.to, eased);
			current.volume.group.quaternion.slerpQuaternions(
				current.rotationFrom,
				current.rotationTo,
				eased,
			);
			current.volume.group.scale.setScalar(
				THREE.MathUtils.lerp(current.scaleFrom, current.scaleTo, eased),
			);
			if (t === 1) {
				motion = undefined;
				current.done();
			} else moving = true;
		}
		if (drawShelf) {
			renderer.render(scene, camera);
			drawShelf = false;
		}
		if (flight) flight.renderer.render(flight.scene, flight.camera);
		if (moving || motion) requestRender();
	};

	const move = (
		volume: Volume,
		to: THREE.Vector3,
		rotationTo: THREE.Quaternion,
		scaleTo: number,
		duration: number,
		done: () => void,
	) => {
		motion = {
			volume,
			from: volume.group.position.clone(),
			to,
			rotationFrom: volume.group.quaternion.clone(),
			rotationTo,
			scaleFrom: volume.group.scale.x,
			scaleTo,
			start: performance.now(),
			duration: reduced.matches ? 1 : duration,
			done,
		};
		requestRender();
	};

	const openBook = (volume: Volume) => {
		if (selected) return;
		selected = volume;
		hovered = undefined;
		dialog.dataset.phase = 'extracting';
		dialog.showModal();
		let flightRenderer: THREE.WebGLRenderer;
		try {
			flightRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
		} catch {
			dialog.close();
			selected = undefined;
			caption.textContent = 'The book could not open. Please try again.';
			return;
		}
		volume.cover.material.map = coverTexture(volume.book, volume.color);
		volume.cover.material.color.set('#ffffff');
		volume.cover.material.needsUpdate = true;
		drawShelf = true;
		flightRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
		flightRenderer.setSize(innerWidth, innerHeight);
		flightRenderer.toneMapping = renderer.toneMapping;
		flightRenderer.toneMappingExposure = renderer.toneMappingExposure;
		flightRenderer.domElement.className = 'book-flight';
		dialog.prepend(flightRenderer.domElement);
		const flightCamera = camera.clone();
		const bounds = canvas.getBoundingClientRect();
		const units = viewHeight / bounds.height;
		flightCamera.left -= bounds.left * units;
		flightCamera.right += (innerWidth - bounds.right) * units;
		flightCamera.top += bounds.top * units;
		flightCamera.bottom -= (innerHeight - bounds.bottom) * units;
		flightCamera.updateProjectionMatrix();
		const flightScene = new THREE.Scene();
		flightScene.add(
			new THREE.HemisphereLight('#fff0ce', '#493b2c', 2.1),
			sun.clone(),
			fill.clone(),
			volume.group,
		);
		flight = { renderer: flightRenderer, camera: flightCamera, scene: flightScene };
		requestRender();
		close.focus();
		title.textContent = volume.book.title;
		iframe = document.createElement('iframe');
		iframe.title = `Sample pages for ${volume.book.title}`;
		iframe.src = `/bookshelf/prototype-reader/?${new URLSearchParams({ book: volume.book.edition.isbn13, color: volume.color })}`;
		mount.replaceChildren(iframe);
		const token = ++generation;
		readerTimeout = window.setTimeout(() => {
			if (token !== generation) return;
			returnBook();
			caption.textContent = 'The reader could not load. Please try again.';
		}, 12000);
	};

	const releaseFlight = () => {
		if (selected) {
			selected.cover.material.map?.dispose();
			selected.cover.material.map = null;
			selected.cover.material.color.set(selected.color);
			selected.cover.material.needsUpdate = true;
		}
		flight?.renderer.domElement.remove();
		flight?.renderer.dispose();
		flight?.renderer.forceContextLoss();
		flight = undefined;
	};

	const finishReturn = () => {
		if (!selected || dialog.dataset.phase === 'returning') return;
		clearTimeout(readerTimeout);
		generation++;
		const volume = selected;
		iframe?.remove();
		iframe = undefined;
		dialog.dataset.phase = 'returning';
		volume.group.visible = true;
		const outside = volume.home.clone();
		outside.z += 1.7;
		move(volume, outside, new THREE.Quaternion(), 1, 300, () => {
			move(volume, volume.home.clone(), new THREE.Quaternion(), 1, 200, () => {
				scene.add(volume.group);
				releaseFlight();
				drawShelf = true;
				selected = undefined;
				dialog.close();
				dialog.dataset.phase = 'shelf';
				volume.button.focus({ preventScroll: true });
			});
		});
	};

	const returnBook = () => {
		if (dialog.dataset.phase === 'closing') return;
		if (dialog.dataset.phase === 'reading') {
			dialog.dataset.phase = 'closing';
			iframe?.contentWindow?.postMessage('close', location.origin);
			readerTimeout = window.setTimeout(finishReturn, 900);
		} else finishReturn();
	};

	addEventListener('message', (event: MessageEvent<unknown>) => {
		if (
			event.origin !== location.origin ||
			event.source !== iframe?.contentWindow ||
			!selected ||
			typeof event.data !== 'object' ||
			event.data === null
		)
			return;
		const data = event.data;
		if (!('type' in data)) return;
		if (data.type === 'reader-closed') {
			finishReturn();
			return;
		}
		if (data.type === 'reader-close') {
			returnBook();
			return;
		}
		if (
			data.type === 'reader-page' &&
			'page' in data &&
			typeof data.page === 'number' &&
			'count' in data &&
			typeof data.count === 'number'
		) {
			pageStatus.textContent =
				data.page === 0
					? 'Cover · Sample book'
					: data.page === data.count - 1
						? 'Back cover'
						: `Page ${data.page} · Sample book`;
			previous.disabled = data.page === 0;
			next.disabled = data.page >= data.count - 1;
		}
		if (
			data.type !== 'reader-ready' ||
			dialog.dataset.phase !== 'extracting' ||
			!('x' in data && 'y' in data && 'width' in data && 'height' in data) ||
			typeof data.x !== 'number' ||
			typeof data.y !== 'number' ||
			typeof data.width !== 'number' ||
			typeof data.height !== 'number'
		)
			return;
		clearTimeout(readerTimeout);
		const volume = selected;
		const frameRect = iframe.getBoundingClientRect();
		const canvasRect = canvas.getBoundingClientRect();
		const centerX = frameRect.x + data.x + data.width / 2;
		const centerY = frameRect.y + data.y + data.height / 2;
		const depth = new THREE.Vector3(0, volume.home.y, 7).project(camera).z;
		const destination = new THREE.Vector3(
			((centerX - canvasRect.x) / canvasRect.width) * 2 - 1,
			1 - ((centerY - canvasRect.y) / canvasRect.height) * 2,
			depth,
		).unproject(camera);
		const orientation = camera.quaternion
			.clone()
			.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2));
		const scale = ((data.height / canvasRect.height) * viewHeight) / volume.height;
		// Align the outer cover plane, rather than the center of the book's thickness.
		destination.addScaledVector(
			new THREE.Vector3(0, 0, 1).applyQuaternion(camera.quaternion),
			(-volume.width / 2) * scale,
		);
		const outside = volume.home.clone();
		outside.z += 1.7;
		const token = generation;
		move(volume, outside, new THREE.Quaternion(), 1, 200, () => {
			if (token !== generation) return;
			move(volume, destination, orientation, scale, 400, () => {
				if (token !== generation) return;
				volume.group.visible = false;
				dialog.dataset.phase = 'reading';
				iframe?.contentWindow?.postMessage('open', location.origin);
				requestRender();
			});
		});
	});
	close.addEventListener('click', returnBook);
	dialog.addEventListener('cancel', (event) => {
		event.preventDefault();
		returnBook();
	});
	previous.addEventListener('click', () =>
		iframe?.contentWindow?.postMessage('previous', location.origin),
	);
	next.addEventListener('click', () => iframe?.contentWindow?.postMessage('next', location.origin));
	dialog.addEventListener('keydown', (event) => {
		if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
			event.preventDefault();
			iframe?.contentWindow?.postMessage(
				event.key === 'ArrowRight' ? 'next' : 'previous',
				location.origin,
			);
		}
	});
	let lastSize = '';
	const resize = new ResizeObserver(() => {
		const size = `${stage.clientWidth}x${stage.clientHeight}`;
		if (lastSize === size) return;
		lastSize = size;
		if (selected) {
			generation++;
			clearTimeout(readerTimeout);
			iframe?.remove();
			iframe = undefined;
			selected.group.visible = true;
			scene.add(selected.group);
			releaseFlight();
			selected = undefined;
			motion = undefined;
			dialog.close();
		}
		layout();
	});
	resize.observe(stage);
	reduced.addEventListener('change', requestRender);
	loading.hidden = true;
	layout();
	addEventListener(
		'pagehide',
		() => {
			cancelAnimationFrame(frame);
			clearTimeout(readerTimeout);
			resize.disconnect();
			iframe?.remove();
			releaseFlight();
			renderer.dispose();
		},
		{ once: true },
	);
}
