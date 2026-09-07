/** Baked glTF assets, two composed camera views, and DOM controls projected into the room. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import furnitureUrl from '../../../assets/room/furniture.glb?url';
import movingUrl from '../../../assets/room/moving.glb?url';
import objectsUrl from '../../../assets/room/objects.glb?url';
import shellUrl from '../../../assets/room/shell.glb?url';

const VIEWS = {
	room: { position: new THREE.Vector3(6.4, 6.5, 9.9), target: new THREE.Vector3(0, 1, -0.2) },
	closer: {
		position: new THREE.Vector3(0.9, 2.4, 4.8),
		target: new THREE.Vector3(0, 0.94, -1.05),
	},
	bookshelf: {
		position: new THREE.Vector3(0.28, 2.05, 2.87),
		target: new THREE.Vector3(-1.12, 0.87, -1.24),
	},
	record: {
		position: new THREE.Vector3(0.98, 1.48, 0.55),
		target: new THREE.Vector3(0.21, 0.8, -1.16),
	},
};
const ANCHORS = {
	bookshelf: new THREE.Vector3(-1.12, 1.03, -1.06),
	record: new THREE.Vector3(0.21, 0.78, -1.16),
};

type View = keyof typeof VIEWS;

/** Mount a room, or leave its poster in place if an asset or WebGL fails. */
export async function mountRoom(
	host: HTMLElement,
	canvas: HTMLCanvasElement,
): Promise<(() => void) | null> {
	const scene = new THREE.Scene();
	const materials = new Set<THREE.Material>();
	const textures = new Set<THREE.Texture>();
	const geometries = new Set<THREE.BufferGeometry>();
	const loader = new GLTFLoader();
	const releaseAssets = () => {
		for (const geometry of geometries) geometry.dispose();
		for (const material of materials) material.dispose();
		for (const texture of textures) {
			texture.dispose();
			if (typeof ImageBitmap !== 'undefined' && texture.source.data instanceof ImageBitmap) {
				texture.source.data.close();
			}
		}
	};
	// Await all loads even after failure so a late success cannot leak GPU resources.
	const results = await Promise.allSettled(
		[shellUrl, furnitureUrl, objectsUrl, movingUrl].map(async (url) => {
			const gltf = await loader.loadAsync(url);
			gltf.scene.traverse((object) => {
				if (!(object instanceof THREE.Mesh)) return;
				geometries.add(object.geometry);
				for (const material of Array.isArray(object.material)
					? object.material
					: [object.material]) {
					materials.add(material);
					if (material instanceof THREE.MeshBasicMaterial && material.map) {
						material.map.anisotropy = 4;
						textures.add(material.map);
					}
				}
			});
			scene.add(gltf.scene);
		}),
	);
	if (results.some((result) => result.status === 'rejected') || !host.isConnected) {
		releaseAssets();
		return null;
	}

	let renderer: THREE.WebGLRenderer;
	try {
		renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			alpha: false,
			powerPreference: 'low-power',
		});
	} catch {
		releaseAssets();
		return null;
	}
	renderer.setClearColor(0xedeae3, 1);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.NoToneMapping;
	renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
	const camera = new THREE.PerspectiveCamera(23.83, 16 / 9, 0.05, 50);
	const position = VIEWS.room.position.clone();
	const target = VIEWS.room.target.clone();
	const fromPosition = position.clone();
	const fromTarget = target.clone();
	const offset = new THREE.Vector2();
	const wantedOffset = new THREE.Vector2();
	const projected = new THREE.Vector3();
	const motion = matchMedia('(prefers-reduced-motion: reduce)');
	const events = new AbortController();
	const hotspots = [...host.querySelectorAll<HTMLElement>('[data-anchor]')];
	let view: View = 'room';
	let alive = true;
	let visible = true;
	let lost = false;
	let frame = 0;
	let transitionStart: number | null = null;
	let lastTime = 0;

	// The platter and the arm are their own nodes in moving.glb, turning about
	// their own axis. Blender's Z became the node's Y in the glTF conversion.
	const platter = scene.getObjectByName('Platter');
	const tonearm = scene.getObjectByName('Tonearm');
	const SPEED = (100 * Math.PI) / 90; // 33 1/3 rpm
	const CUED = -0.1846; // the headshell reaches the lead-in groove
	const DROP = 0.1; // and noses down onto it
	let cueing = false;
	let cue = 0; // 0 parked, 1 on the record
	let spin = 0; // rad/s, spinning up and down like a real platter

	function turntable(dt: number) {
		const wanted = cueing ? 1 : 0;
		if (motion.matches) {
			// A record turning forever is exactly what reduced motion asks to stop.
			cue = wanted;
			spin = 0;
		} else {
			cue = THREE.MathUtils.damp(cue, wanted, 4, dt);
			if (Math.abs(cue - wanted) < 0.001) cue = wanted;
			spin = THREE.MathUtils.damp(spin, cueing ? SPEED : 0, 2.5, dt);
			if (Math.abs(spin - (cueing ? SPEED : 0)) < 0.01) spin = cueing ? SPEED : 0;
			if (platter) platter.rotation.y -= spin * dt;
		}
		if (tonearm) {
			tonearm.rotation.y = CUED * cue;
			tonearm.rotation.x = DROP * cue;
		}
		return spin !== 0 || cue !== wanted;
	}

	function draw() {
		camera.position.copy(position);
		camera.position.x += offset.x;
		camera.position.y += offset.y;
		camera.lookAt(target);
		camera.updateMatrixWorld();
		renderer.render(scene, camera);
		for (const hotspot of hotspots) {
			const name = hotspot.dataset.anchor;
			if (name !== 'bookshelf' && name !== 'record') continue;
			projected.copy(ANCHORS[name]).project(camera);
			// A close view leaves the other object off frame: take its ring out of
			// the picture and out of the tab order rather than parking it outside.
			// The wide mix credit needs more room than a ring before it fits.
			const slack = hotspot.classList.contains('living-room__mix') ? 0.6 : 0.98;
			hotspot.hidden =
				projected.z > 1 || Math.abs(projected.x) > slack || Math.abs(projected.y) > slack;
			hotspot.style.left = `${(projected.x * 0.5 + 0.5) * 100}%`;
			hotspot.style.top = `${(-projected.y * 0.5 + 0.5) * 100}%`;
		}
		host.dataset.placed = '';
	}

	function tick(now: number) {
		frame = 0;
		if (!alive || lost || !visible || document.hidden) return;
		const dt = Math.min((now - lastTime) / 1000, 0.05);
		lastTime = now;
		if (transitionStart !== null) {
			const t = motion.matches ? 1 : Math.min((now - transitionStart) / 1200, 1);
			const eased = t < 0.5 ? 16 * t ** 5 : 1 - (-2 * t + 2) ** 5 / 2;
			position.lerpVectors(fromPosition, VIEWS[view].position, eased);
			target.lerpVectors(fromTarget, VIEWS[view].target, eased);
			if (t === 1) transitionStart = null;
		}
		offset.lerp(wantedOffset, motion.matches ? 1 : 1 - Math.exp(-8 * dt));
		if (offset.distanceToSquared(wantedOffset) < 0.000001) offset.copy(wantedOffset);
		const moving = turntable(dt);
		draw();
		if (transitionStart !== null || moving || !offset.equals(wantedOffset)) requestDraw();
	}

	function requestDraw() {
		if (!frame && alive && visible && !lost && !document.hidden)
			frame = requestAnimationFrame(tick);
	}

	function changeView(next: View) {
		if (next === view) return;
		fromPosition.copy(position);
		fromTarget.copy(target);
		view = next;
		transitionStart = performance.now();
		wantedOffset.set(0, 0);
		publish(view);
		if (shelf) {
			shelf.textContent = view === 'bookshelf' ? 'Open the bookshelf' : 'Browse the bookshelf';
			// Refresh the caption if the pointer or keyboard is already on it.
			if (shelf.matches(':hover, :focus-visible'))
				shelf.dispatchEvent(new Event('pointerover', { bubbles: true }));
		}
		requestDraw();
	}

	// Clicking an object travels to it. A link travels first and navigates on the
	// second press, so the trip somewhere else is never a surprise.
	const shelf = hotspots.find(
		(hotspot) => hotspot.dataset.anchor === 'bookshelf' && hotspot instanceof HTMLAnchorElement,
	);
	for (const hotspot of hotspots) {
		if (hotspot.classList.contains('living-room__mix')) continue;
		const anchor = hotspot.dataset.anchor;
		if (anchor !== 'bookshelf' && anchor !== 'record') continue;
		hotspot.addEventListener(
			'click',
			(event) => {
				const plain = event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey;
				if (hotspot === shelf && view !== 'bookshelf' && plain) event.preventDefault();
				changeView(anchor);
			},
			{ signal: events.signal },
		);
	}

	// The DOM owns playback; the room follows the state it publishes.
	const playing = new MutationObserver(() => {
		cueing = host.hasAttribute('data-playing');
		lastTime = performance.now();
		requestDraw();
	});
	playing.observe(host, { attributeFilter: ['data-playing'] });

	function publish(current: View) {
		host.dataset.view = current;
		canvas.setAttribute(
			'aria-label',
			current === 'room' ? 'Step closer into the room' : 'Step back to the whole room',
		);
	}

	// Clicking the room itself steps in, and steps back out from any close view.
	// The canvas is the control, so the keyboard reaches it without any chrome.
	const step = () => changeView(view === 'room' ? 'closer' : 'room');
	// Focusable only while the scene is actually there to steer: an invisible
	// canvas must not keep a tab stop.
	function arm(on: boolean) {
		if (on) {
			canvas.setAttribute('tabindex', '0');
			canvas.setAttribute('role', 'button');
			canvas.removeAttribute('aria-hidden');
		} else {
			canvas.removeAttribute('tabindex');
			canvas.removeAttribute('role');
			canvas.setAttribute('aria-hidden', 'true');
		}
	}
	arm(true);
	canvas.addEventListener('click', step, { signal: events.signal });
	canvas.addEventListener(
		'keydown',
		(event) => {
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				step();
			}
		},
		{ signal: events.signal },
	);
	// Escape backs out while the room is the thing on screen, wherever focus is.
	document.addEventListener(
		'keydown',
		(event) => {
			if (event.key !== 'Escape' || view === 'room' || !visible) return;
			changeView('room');
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'pointermove',
		(event) => {
			if (motion.matches || event.pointerType !== 'mouse') return;
			const bounds = host.getBoundingClientRect();
			wantedOffset.set(
				((event.clientX - bounds.left) / bounds.width - 0.5) * 0.12,
				(0.5 - (event.clientY - bounds.top) / bounds.height) * 0.06,
			);
			requestDraw();
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'pointerleave',
		() => {
			wantedOffset.set(0, 0);
			requestDraw();
		},
		{ signal: events.signal },
	);
	motion.addEventListener(
		'change',
		() => {
			wantedOffset.set(0, 0);
			requestDraw();
		},
		{ signal: events.signal },
	);

	const resize = new ResizeObserver(() => {
		const width = host.clientWidth;
		const height = canvas.clientHeight;
		if (!width || !height) return;
		renderer.setSize(width, height, false);
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		requestDraw();
	});
	resize.observe(host);
	const intersection = new IntersectionObserver(([entry]) => {
		visible = entry?.isIntersecting ?? false;
		if (visible) requestDraw();
		else {
			cancelAnimationFrame(frame);
			frame = 0;
		}
	});
	intersection.observe(host);
	document.addEventListener(
		'visibilitychange',
		() => {
			if (document.hidden) {
				cancelAnimationFrame(frame);
				frame = 0;
			} else requestDraw();
		},
		{ signal: events.signal },
	);
	canvas.addEventListener(
		'webglcontextlost',
		(event) => {
			event.preventDefault();
			lost = true;
			arm(false);
			delete host.dataset.live;
			delete host.dataset.placed;
			for (const hotspot of hotspots) {
				hotspot.hidden = false;
				hotspot.style.left = '';
				hotspot.style.top = '';
			}
			cancelAnimationFrame(frame);
			frame = 0;
		},
		{ signal: events.signal },
	);
	canvas.addEventListener(
		'webglcontextrestored',
		() => {
			lost = false;
			arm(true);
			requestDraw();
			host.dataset.live = '';
		},
		{ signal: events.signal },
	);

	function dispose() {
		if (!alive) return;
		alive = false;
		cancelAnimationFrame(frame);
		events.abort();
		resize.disconnect();
		intersection.disconnect();
		playing.disconnect();
		releaseAssets();
		renderer.dispose();
		renderer.forceContextLoss();
		arm(false);
		delete host.dataset.live;
	}

	addEventListener(
		'pagehide',
		(event) => {
			if (!event.persisted) dispose();
		},
		{ signal: events.signal },
	);
	renderer.setSize(host.clientWidth, canvas.clientHeight, false);
	publish('room');
	draw();
	return dispose;
}
