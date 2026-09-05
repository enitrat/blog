import * as THREE from 'three';
import { type Book, books } from '../../../booksData';
import { pageRangeFor, pleiadeStyleFor, spineWidthFor } from '../../../utils/pleiade';
import { PLEIADE_HEX, paintBackCover, paintCover, paintGilt, paintSpine } from './pleiade-paint';

type Volume = {
	book: Book;
	group: THREE.Group;
	home: THREE.Vector3;
	button: HTMLButtonElement;
	width: number;
	height: number;
	color: string;
	coverMaterial: THREE.MeshBasicMaterial;
	shelfCover: THREE.CanvasTexture;
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

const PITCH = 2.36;
const DEPTH = 1.28;
const BREAKPOINT = 720;

function texture(
	width: number,
	height: number,
	draw: (ctx: CanvasRenderingContext2D) => void,
	sharp = false,
) {
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('Canvas 2D is required to draw book textures.');
	draw(ctx);
	const result = new THREE.CanvasTexture(canvas);
	result.colorSpace = THREE.SRGBColorSpace;
	if (sharp) {
		result.generateMipmaps = false;
		result.minFilter = THREE.LinearFilter;
		result.magFilter = THREE.LinearFilter;
	} else result.anisotropy = 4;
	return result;
}

/** Mount the disposable library study. The scene renders only while a book moves. */
export async function startLibrary() {
	const stage = document.getElementById('library-stage');
	const viewport = document.getElementById('library-viewport');
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
		!viewport ||
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
	const camera = new THREE.OrthographicCamera(-10, 10, 8, -8, 0.1, 80);
	const layoutCamera = new THREE.OrthographicCamera(-10, 10, 8, -8, 0.1, 80);
	scene.add(new THREE.HemisphereLight('#fff0ce', '#493b2c', 2.1));
	const sun = new THREE.DirectionalLight('#ffe3af', 2.8);
	scene.add(sun);
	const fill = new THREE.DirectionalLight('#eee8dd', 0.7);
	scene.add(fill);

	const oakSize = 1024;
	const oakTexture = texture(oakSize, oakSize, (ctx) => {
		ctx.fillStyle = '#65482f';
		ctx.fillRect(0, 0, oakSize, oakSize);
		for (let i = 0; i < 1200; i++) {
			const seed = Math.sin(i * 127.1) * 43758.5453;
			const noise = seed - Math.floor(seed);
			ctx.strokeStyle =
				i % 3 ? `rgba(34,18,6,${0.03 + noise * 0.13})` : `rgba(230,177,103,${noise * 0.18})`;
			ctx.lineWidth = 0.4 + noise * 1.8;
			ctx.beginPath();
			for (let x = 0; x <= oakSize; x += 8) {
				const y = noise * oakSize + Math.sin(x / 88 + i * 0.6) * 3.2 + Math.sin(x / 220 + i) * 6;
				if (x === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.stroke();
		}
	});
	oakTexture.wrapS = THREE.RepeatWrapping;
	oakTexture.wrapT = THREE.RepeatWrapping;
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
	const giltMap = texture(256, 64, paintGilt);
	const giltMat = new THREE.MeshBasicMaterial({ map: giltMap });
	const boardMat = new THREE.MeshBasicMaterial({ color: '#241910' });
	const backByColor = new Map<string, THREE.MeshBasicMaterial>();
	const pageRange = pageRangeFor(books);

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
	let mobile = false;
	let viewHeight = 12;
	let viewWidth = 12;
	let caseWidth = 6;
	let caseHeight = 12;
	let eyeX = 1.9;
	let eyeY = 1.7;
	let eyeZ = 15;

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

	const lookAtY = (cam: THREE.OrthographicCamera, targetY: number) => {
		cam.position.set(eyeX, targetY + eyeY, eyeZ);
		cam.lookAt(0, targetY, 0);
	};

	for (const [index, book] of books.entries()) {
		const style = pleiadeStyleFor(book.author);
		const color = PLEIADE_HEX[style.color];
		const px = spineWidthFor(book.edition.pageCount, pageRange, 'extended');
		const width = 0.5 + ((px - 58) / 60) * 0.3;
		const height = 2.08 + (index % 3) * 0.03;
		const group = new THREE.Group();
		const painted = { author: style.label, title: book.title, color };
		const spineMap = texture(256, 1024, (ctx) => paintSpine(ctx, painted), true);
		const shelfCover = texture(
			256,
			384,
			(ctx) => paintCover(ctx, { author: book.author, title: book.title, color }),
			true,
		);
		let backMat = backByColor.get(color);
		if (!backMat) {
			backMat = new THREE.MeshBasicMaterial({
				map: texture(192, 288, (ctx) => paintBackCover(ctx, color), true),
			});
			backByColor.set(color, backMat);
		}
		const spineMat = new THREE.MeshBasicMaterial({ map: spineMap });
		const coverMaterial = new THREE.MeshBasicMaterial({ map: shelfCover });
		const body = new THREE.Mesh(new THREE.BoxGeometry(width, height, DEPTH), [
			coverMaterial,
			backMat,
			giltMat,
			giltMat,
			spineMat,
			boardMat,
		]);
		group.add(body);
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
			coverMaterial,
			shelfCover,
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

	const frameCamera = () => {
		const canvasH = viewport.clientHeight;
		const travel = Math.max(0, caseHeight - viewHeight);
		const maxScroll = Math.max(1, stage.offsetHeight - canvasH);
		const progress = Math.min(1, Math.max(0, -stage.getBoundingClientRect().top / maxScroll));
		const lookY = travel > 0 ? caseHeight - viewHeight / 2 - progress * travel : caseHeight / 2;
		camera.left = -viewWidth / 2;
		camera.right = viewWidth / 2;
		camera.top = viewHeight / 2;
		camera.bottom = -viewHeight / 2;
		lookAtY(camera, lookY);
		camera.updateProjectionMatrix();
		camera.updateMatrixWorld();
		sun.position.set(-6, lookY + 11, 12);
		fill.position.set(7, lookY + 4, 3);
		drawShelf = true;
	};

	const layout = () => {
		drawShelf = true;
		mobile = viewport.clientWidth < BREAKPOINT;
		const columns = mobile ? 4 : 8;
		const rows = Math.ceil(volumes.length / columns);
		const gap = mobile ? 0.02 : 0.022;
		let packedWidth = 0;
		for (let row = 0; row < rows; row++) {
			const rowVolumes = volumes.slice(row * columns, row * columns + columns);
			const packed = rowVolumes.reduce((sum, volume) => sum + volume.width, 0);
			packedWidth = Math.max(packedWidth, packed + gap * Math.max(0, rowVolumes.length - 1));
		}
		caseWidth = packedWidth + 0.7;
		caseHeight = rows * PITCH + 0.62;
		eyeX = mobile ? 0.95 : 1.45;
		eyeY = mobile ? 1.05 : 1.35;
		eyeZ = mobile ? 11 : 14;
		const inner = packedWidth + 0.08;
		const innerLeft = -inner / 2;
		const innerRight = inner / 2;
		for (const child of [...caseGroup.children]) {
			if (child instanceof THREE.Mesh) child.geometry.dispose();
			caseGroup.remove(child);
		}
		box(caseWidth, caseHeight, 0.14, 0, caseHeight / 2, -0.8, darkOak);
		for (let row = 0; row < rows; row++) {
			box(caseWidth - 0.3, PITCH - 0.28, 0.03, 0, row * PITCH + 1.32, -0.7, backMaterial);
		}
		for (let row = 0; row <= rows; row++) {
			box(caseWidth + 0.1, 0.2, 1.92, 0, row * PITCH + 0.25, 0.05, oak);
			box(caseWidth + 0.16, 0.055, 0.07, 0, row * PITCH + 0.32, 1.04, trim);
			box(caseWidth, 0.08, 0.1, 0, row * PITCH + 0.15, 1.01, darkOak);
		}
		for (const x of [-caseWidth / 2, caseWidth / 2]) {
			box(0.24, caseHeight, 1.98, x, caseHeight / 2, 0.05, verticalOak);
			box(0.08, caseHeight - 0.1, 0.055, x, caseHeight / 2, 1.06, trim);
		}
		box(caseWidth + 0.45, 0.2, 2.1, 0, caseHeight + 0.04, 0.03, oak);
		box(caseWidth + 0.3, 0.1, 2.02, 0, caseHeight - 0.12, 0.03, trim);
		box(caseWidth + 0.3, 0.32, 2.06, 0, 0.02, 0.03, oak);

		for (let row = 0; row < rows; row++) {
			const rowVolumes = volumes.slice(row * columns, row * columns + columns);
			const packed = rowVolumes.reduce((sum, volume) => sum + volume.width, 0);
			const total = packed + gap * Math.max(0, rowVolumes.length - 1);
			let x = (innerLeft + innerRight) / 2 - total / 2;
			const shelfY = (rows - 1 - row) * PITCH;
			for (const volume of rowVolumes) {
				x += volume.width / 2;
				volume.home.set(x, shelfY + 0.36 + volume.height / 2, 0.18);
				volume.group.position.copy(volume.home);
				volume.group.quaternion.identity();
				volume.group.scale.setScalar(1);
				x += volume.width / 2 + gap;
			}
		}

		const aspect = viewport.clientWidth / Math.max(1, viewport.clientHeight);
		viewWidth = caseWidth + (mobile ? 0.28 : 0.52);
		viewHeight = viewWidth / aspect;
		const minHeight = PITCH * (mobile ? 1.28 : 1.38) + 0.28;
		if (viewHeight < minHeight) {
			viewHeight = minHeight;
			viewWidth = viewHeight * aspect;
		}
		const minWidth = caseWidth + (mobile ? 0.22 : 0.4);
		if (viewWidth < minWidth) {
			viewWidth = minWidth;
			viewHeight = viewWidth / aspect;
		}

		const canvasH = viewport.clientHeight;
		stage.style.height = `${Math.max(canvasH, Math.round(canvasH * (caseHeight / viewHeight)))}px`;
		renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);

		layoutCamera.left = -viewWidth / 2;
		layoutCamera.right = viewWidth / 2;
		layoutCamera.top = caseHeight / 2;
		layoutCamera.bottom = -caseHeight / 2;
		lookAtY(layoutCamera, caseHeight / 2);
		layoutCamera.updateProjectionMatrix();
		layoutCamera.updateMatrixWorld();

		const stageW = stage.clientWidth;
		const stageH = stage.clientHeight;
		const spine = new THREE.Vector3();
		for (const volume of volumes) {
			spine.set(volume.home.x, volume.home.y, volume.home.z + DEPTH / 2).project(layoutCamera);
			const w = (volume.width / viewWidth) * stageW;
			const h = (volume.height / caseHeight) * stageH;
			Object.assign(volume.button.style, {
				left: `${((spine.x + 1) / 2) * stageW - w / 2}px`,
				top: `${((1 - spine.y) / 2) * stageH - h / 2}px`,
				width: `${Math.max(w, 44)}px`,
				height: `${h}px`,
			});
		}
		frameCamera();
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
		volume.coverMaterial.map = texture(
			512,
			768,
			(ctx) =>
				paintCover(ctx, {
					author: volume.book.author,
					title: volume.book.title,
					color: volume.color,
				}),
			true,
		);
		volume.coverMaterial.needsUpdate = true;
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
			const map = selected.coverMaterial.map;
			if (map && map !== selected.shelfCover) map.dispose();
			selected.coverMaterial.map = selected.shelfCover;
			selected.coverMaterial.needsUpdate = true;
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
	const relayout = () => {
		const size = `${viewport.clientWidth}x${viewport.clientHeight}`;
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
	};
	const resize = new ResizeObserver(relayout);
	resize.observe(viewport);
	const onScroll = () => {
		frameCamera();
		requestRender();
	};
	addEventListener('scroll', onScroll, { passive: true });
	visualViewport?.addEventListener('resize', relayout);
	reduced.addEventListener('change', requestRender);
	loading.hidden = true;
	relayout();
	addEventListener(
		'pagehide',
		() => {
			cancelAnimationFrame(frame);
			clearTimeout(readerTimeout);
			resize.disconnect();
			removeEventListener('scroll', onScroll);
			visualViewport?.removeEventListener('resize', relayout);
			iframe?.remove();
			releaseFlight();
			renderer.dispose();
		},
		{ once: true },
	);
}
