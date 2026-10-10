import { useEffect, useMemo, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Project, ViewerProps } from '../types';
import {pixelsPerMeter} from '../viewer/poseMath';
import {preservePhotoTextureColors,preserveTextureDetail} from '../viewer/materialPolicy';
import { applyOcclusionVisibility, CANONICAL_BRIDGE, configureFrameShadows, createFaceOccluder, createSkullOccluder, FACE_ANCHORS, parseFaceTopology, updateFaceOccluder } from '../viewer/occlusion';
import { defaultARFit, readARFit, saveARFit } from '../viewer/arFit';
import './viewer.css';

type Quality = 'low' | 'balanced' | 'high';
type SurfaceDisplay = 'photo' | 'material' | 'clay' | 'silhouette';
type PreviewFit = 'cover' | 'contain';
type InspectScene = {
  setWireframe: (enabled: boolean) => void;
  setVisibleParts: (parts: Set<string>) => void;
  setView: (view: 'front' | 'side' | 'other-side' | 'top' | 'reset') => void;
  setLighting: (intensity: number) => void;
};
type Dimensions = { width: number; height: number; depth: number };
type Landmark = { x: number; y: number; z: number };
type FaceMatrix = { data: number[] | Float32Array };
type ARSettings = { pd: number; scale: number; horizontal: number; vertical: number; depth: number; previewFit: PreviewFit; headWidth: number; headHeight: number; headHorizontal: number; headVertical: number; skullDepth: number; showSkull: boolean; showFaceOccluder: boolean; showAnchors: boolean; showGlassesWireframe: boolean; softShadow: boolean; useFaceOcclusion: boolean; useSkullOcclusion: boolean };

const mm = (meters: number) => `${(meters * 1000).toFixed(1)} mm`;
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const frontPhoto = (project: Project) => project.photos.find(photo => photo.view === 'front' && photo.role === 'fitting')
  ?? project.photos.find(photo => photo.view.includes('front') && photo.role === 'fitting')
  ?? project.photos.find(photo => photo.role === 'fitting');

function disposeObject(root: THREE.Object3D) {
  root.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return;
    node.geometry?.dispose();
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    materials.forEach(material => {
      if (!material) return;
      Object.values(material).forEach(value => {
        if (value instanceof THREE.Texture) value.dispose();
      });
      material.dispose();
    });
  });
}

function getDimensions(root: THREE.Object3D): Dimensions {
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  return { width: size.x, height: size.y, depth: size.z };
}

function qualityPixelRatio(quality: Quality) {
  return Math.min(window.devicePixelRatio || 1, quality === 'low' ? 1 : quality === 'balanced' ? 1.5 : 2);
}

function makeRenderer(canvas: HTMLCanvasElement, quality: Quality, alpha = false) {
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha, antialias: quality !== 'low', powerPreference: quality === 'high' ? 'high-performance' : 'default' });
  } catch {
    throw new Error('WebGL is unavailable. Enable hardware graphics support or use another browser.');
  }
  renderer.setPixelRatio(qualityPixelRatio(quality));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.5;
  return renderer;
}

function Inspection({ project, quality }: { project: Project; quality: Quality }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const sceneApi = useRef<InspectScene | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [surfaceDisplay, setSurfaceDisplay] = useState<SurfaceDisplay>('photo');
  const [parts, setParts] = useState<string[]>([]);
  const [visibleParts, setVisibleParts] = useState<Set<string>>(new Set());
  const [dimensions, setDimensions] = useState<Dimensions | null>(null);
  const [comparison, setComparison] = useState<'off' | 'overlay' | 'split'>('off');
  const [blend, setBlend] = useState(50);
  const [selectedPhotoId, setSelectedPhotoId] = useState<string | undefined>(() => frontPhoto(project)?.id);
  const [photoZoom, setPhotoZoom] = useState(100);
  const [photoX, setPhotoX] = useState(0);
  const [photoY, setPhotoY] = useState(0);
  const [photoAngle, setPhotoAngle] = useState(0);
  const [lighting, setLighting] = useState(100);
  const source = project.photos.find(photo => photo.id === selectedPhotoId) ?? frontPhoto(project);
  const modelUrl = project.artifact?.model_url;

  useEffect(() => {
    const element = canvas.current;
    if (!element || !modelUrl) return;
    let alive = true;
    let frame = 0;
    let gltfRoot: THREE.Object3D | null = null;
    setError('');
    setLoading(true);
    setParts([]);
    setDimensions(null);
    let renderer: THREE.WebGLRenderer;
    try { renderer = makeRenderer(element, quality); }
    catch (error) { setLoading(false); setError(errorText(error)); return; }
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#ffffff');
    const camera = new THREE.PerspectiveCamera(38, 1, 0.001, 100);
    camera.position.set(0.12, 0.055, 0.32);
    const controls = new OrbitControls(camera, element);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.045;
    controls.maxDistance = 2;
    scene.add(new THREE.HemisphereLight('#ffffff', '#adc0b6', 2.4));
    const key = new THREE.DirectionalLight('#ffffff', 3);
    key.position.set(0.3, 0.4, 0.5);
    scene.add(key);
    const rim = new THREE.DirectionalLight('#b7eede', 1.5);
    rim.position.set(-0.2, 0.1, -0.4);
    scene.add(rim);
    const resize = () => {
      const width = element.clientWidth;
      const height = element.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    const render = () => {
      frame = requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();
    new GLTFLoader().load(modelUrl, gltf => {
      if (!alive) { disposeObject(gltf.scene); return; }
      gltfRoot = gltf.scene;
      preserveTextureDetail(gltfRoot, renderer);
      if (surfaceDisplay === 'photo') preservePhotoTextureColors(gltfRoot);
      if (surfaceDisplay === 'clay' || surfaceDisplay === 'silhouette') {
        gltfRoot.traverse(node => {
          if (!(node instanceof THREE.Mesh)) return;
          const original = Array.isArray(node.material) ? node.material : [node.material];
          const replacement = original.map(() => surfaceDisplay === 'clay'
            ? new THREE.MeshStandardMaterial({ color: '#b8bdba', roughness: 1, metalness: 0, side: THREE.DoubleSide, flatShading: !node.geometry.getAttribute('normal') })
            : new THREE.MeshBasicMaterial({ color: '#111111', side: THREE.DoubleSide, toneMapped: false }));
          node.material = Array.isArray(node.material) ? replacement : replacement[0];
          original.forEach(material => material.dispose());
        });
      }
      scene.add(gltfRoot);
      const box = new THREE.Box3().setFromObject(gltfRoot);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const span = Math.max(size.x, size.y, size.z, 0.05);
      controls.target.copy(center);
      // Start at the source front angle. An elevated perspective can project
      // rear arms above the brow and misleadingly resemble inverted geometry.
      camera.position.copy(center).add(new THREE.Vector3(0, 0, span * 1.9));
      camera.near = Math.max(0.0001, span / 1000);
      camera.far = Math.max(10, span * 100);
      camera.updateProjectionMatrix();
      controls.minDistance = span * 0.25;
      controls.maxDistance = span * 10;
      controls.update();
      const names = new Set<string>();
      gltfRoot.traverse(node => { if (node instanceof THREE.Mesh) names.add(node.name || 'Unnamed part'); });
      const list = [...names];
      setParts(list);
      setVisibleParts(new Set(list));
      setDimensions(getDimensions(gltfRoot));
      sceneApi.current = {
        setWireframe(enabled) {
          gltfRoot?.traverse(node => {
            if (!(node instanceof THREE.Mesh)) return;
            (Array.isArray(node.material) ? node.material : [node.material]).forEach(material => {
              if (material && 'wireframe' in material) (material as THREE.MeshStandardMaterial).wireframe = enabled;
            });
          });
        },
        setVisibleParts(enabled) {
          gltfRoot?.traverse(node => { if (node instanceof THREE.Mesh) node.visible = enabled.has(node.name || 'Unnamed part'); });
        },
        setView(view) {
          // Flush any orbit damping before applying a reproducible preset.
          const damping = controls.enableDamping;
          controls.enableDamping = false;
          controls.update();
          const d = span * 1.9;
          const offset = view === 'side' ? new THREE.Vector3(d, 0, 0)
            : view === 'other-side' ? new THREE.Vector3(-d, 0, 0)
            // A perfectly vertical view is singular with Y-up OrbitControls;
            // a tiny fixed Z offset avoids arbitrary 90-degree camera rolls.
            : view === 'top' ? new THREE.Vector3(0, d, d * 0.0001)
            : view === 'reset' ? new THREE.Vector3(d * 0.35, d * 0.15, d)
            : new THREE.Vector3(0, 0, d);
          camera.position.copy(center).add(offset);
          camera.up.set(0, 1, 0);
          camera.lookAt(center);
          controls.update();
          controls.enableDamping = damping;
        },
        setLighting(intensity) { key.intensity = 3 * intensity; rim.intensity = 1.5 * intensity; },
      };
      setLoading(false);
    }, undefined, err => {
      if (!alive) return;
      setLoading(false);
      setError(`Could not load the exported GLB: ${errorText(err)}`);
    });
    return () => {
      alive = false;
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      sceneApi.current = null;
      if (gltfRoot) disposeObject(gltfRoot);
      renderer.dispose();
    };
  }, [modelUrl, quality, surfaceDisplay]);

  useEffect(() => { sceneApi.current?.setWireframe(wireframe); }, [wireframe, parts]);
  useEffect(() => { sceneApi.current?.setVisibleParts(visibleParts); }, [visibleParts]);
  useEffect(() => { sceneApi.current?.setLighting(lighting / 100); }, [lighting, parts]);
  const togglePart = (name: string) => setVisibleParts(current => {
    const next = new Set(current);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });

  return <div className="mv-workspace">
    <div className="mv-stage-panel">
      <div className={`mv-inspect-stage mv-compare-${comparison}`}>
        <canvas ref={canvas} aria-label="Interactive three-dimensional eyewear model" />
        {source && comparison !== 'off' && <div className="mv-source-layer" style={{ opacity: comparison === 'overlay' ? blend / 100 : 1 }}><img src={source.url} alt={`${source.view} ${source.role} photograph`} style={{ transform: `translate(${photoX}px, ${photoY}px) rotate(${photoAngle}deg) scale(${photoZoom / 100})` }} /></div>}
        {comparison === 'split' && <div className="mv-split-labels"><span>Source photo</span><span>3D render</span></div>}
        {loading && <div className="mv-stage-status">Loading exported model…</div>}
        {error && <div className="mv-stage-status mv-error">{error}</div>}
        {!source && <div className="mv-photo-note">Add a fitting or validation photo to compare the render.</div>}
      </div>
      <div className="mv-toolbar" aria-label="Model controls">
        <div className="mv-button-group">
          <button type="button" onClick={() => sceneApi.current?.setView('front')}>Front</button>
          <button type="button" onClick={() => sceneApi.current?.setView('side')}>Side</button>
          <button type="button" onClick={() => sceneApi.current?.setView('other-side')}>Other side</button>
          <button type="button" onClick={() => sceneApi.current?.setView('top')}>Top</button>
          <button type="button" onClick={() => sceneApi.current?.setView('reset')}>Orbit</button>
        </div>
        <label className="mv-check"><input type="checkbox" checked={wireframe} onChange={event => setWireframe(event.target.checked)} /> Wireframe</label>
        <label className="mv-select-label">Surface display <select value={surfaceDisplay} onChange={event => setSurfaceDisplay(event.target.value as SurfaceDisplay)}><option value="photo">Photo colors</option><option value="material">Exported material</option><option value="clay">Neutral clay</option><option value="silhouette">Silhouette</option></select></label>
      </div>
      <p className="mv-muted">{surfaceDisplay === 'photo' ? 'Photo-color preview and AR preserve photographed frame textures without studio relighting. This does not recover true calibrated material color.' : surfaceDisplay === 'material' ? 'Studio lighting changes perceived color and brightness. The exported GLB retains its lit materials; AR uses photo-color rendering for textured parts.' : surfaceDisplay === 'clay' ? 'Neutral clay hides texture maps while retaining geometry-based lighting. Missing exported normals use flat shading.' : 'Silhouette hides textures and lighting so the physical outline can be inspected.'}</p>
      <div className="mv-compare-controls">
        <span>Source comparison</span>
        <label className="mv-select-label">Photo <select value={source?.id ?? ''} disabled={!source} onChange={event => { setSelectedPhotoId(event.target.value); setPhotoZoom(100); setPhotoX(0); setPhotoY(0); setPhotoAngle(0); }}><option value="" disabled>Choose photo</option>{project.photos.map(photo => <option key={photo.id} value={photo.id}>{photo.view} · {photo.role}{photo.role === 'validation' ? ' (withheld)' : ''}</option>)}</select></label>
        <div className="mv-button-group">
          {(['off', 'overlay', 'split'] as const).map(value => <button type="button" className={comparison === value ? 'active' : ''} key={value} disabled={!source} onClick={() => { setComparison(value); sceneApi.current?.setView('front'); }}>{value === 'off' ? 'Off' : value === 'overlay' ? 'Overlay' : 'Split'}</button>)}
        </div>
        {comparison === 'overlay' && <label className="mv-range-label">Photo opacity <input type="range" min="0" max="100" value={blend} onChange={event => setBlend(Number(event.target.value))} /> {blend}%</label>}
      </div>
      {comparison !== 'off' && <div className="mv-alignment-controls"><span>Align photo to render</span><label>Zoom <input type="range" min="50" max="250" value={photoZoom} onChange={event => setPhotoZoom(Number(event.target.value))} />{photoZoom}%</label><label>Horizontal <input type="range" min="-300" max="300" value={photoX} onChange={event => setPhotoX(Number(event.target.value))} />{photoX}px</label><label>Vertical <input type="range" min="-300" max="300" value={photoY} onChange={event => setPhotoY(Number(event.target.value))} />{photoY}px</label><label>Rotate <input type="range" min="-45" max="45" value={photoAngle} onChange={event => setPhotoAngle(Number(event.target.value))} />{photoAngle}°</label><button type="button" onClick={() => { setPhotoZoom(100); setPhotoX(0); setPhotoY(0); setPhotoAngle(0); }}>Reset photo</button></div>}
      {comparison !== 'off' && <p className="mv-muted mv-comparison-caveat">Manually align the photo with zoom, position, and rotation; drag the 3D render to match its angle. No calibrated silhouette or landmark score is available for this view.</p>}
    </div>
    <aside className="mv-sidebar">
      <section className="mv-card"><h3>Model dimensions</h3>
        {dimensions ? <dl className="mv-metrics"><div><dt>Width</dt><dd>{mm(dimensions.width)}</dd></div><div><dt>Height</dt><dd>{mm(dimensions.height)}</dd></div><div><dt>Depth including temples</dt><dd>{mm(dimensions.depth)}</dd></div></dl> : <p className="mv-muted">Available after the GLB loads.</p>}
        <p className="mv-muted">GLB bounding box in meters, shown in millimeters. This checks the exported geometry; it does not verify fit on a person.</p>
      </section>
      <section className="mv-card"><h3>Parts</h3>
        {parts.length ? <div className="mv-parts">{parts.map(name => <label key={name}><input type="checkbox" checked={visibleParts.has(name)} onChange={() => togglePart(name)} /><span>{name}</span></label>)}</div> : <p className="mv-muted">Named mesh parts appear here.</p>}
      </section>
      <section className="mv-card"><h3>Lighting</h3><label className="mv-field">Studio light <span>{lighting}%</span><input type="range" min="25" max="200" step="5" value={lighting} onChange={event => setLighting(Number(event.target.value))} /></label></section>
      <section className="mv-card"><h3>Evidence</h3><Evidence project={project} /></section>
    </aside>
  </div>;
}

function Evidence({ project }: { project: Project }) {
  const metadata = project.artifact?.metadata ?? {};
  const supported: { name: string; value: number | string; unit?: string }[] = Array.isArray(metadata.metrics) ? metadata.metrics.filter((metric: unknown): metric is { name: string; value: number | string; unit?: string } => {
    if (!metric || typeof metric !== 'object') return false;
    const item = metric as Record<string, unknown>;
    return typeof item.name === 'string' && (typeof item.value === 'number' || typeof item.value === 'string') && (item.unit === undefined || typeof item.unit === 'string');
  }) : [];
  const warnings: string[] = Array.isArray(metadata.warnings) ? metadata.warnings.filter((warning: unknown): warning is string => typeof warning === 'string') : [];
  return <>
    <p className="mv-muted">Measurements and their provenance:</p>
    <dl className="mv-evidence-list">{Object.entries(project.measurements).map(([key, measurement]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{measurement.value} mm <small>{measurement.source}</small></dd></div>)}</dl>
    {supported.length > 0 && <><h4>Reported metrics</h4><dl className="mv-evidence-list">{supported.map((metric, index) => <div key={`${metric.name}-${index}`}><dt>{metric.name}</dt><dd>{metric.value}{metric.unit ? ` ${metric.unit}` : ''}</dd></div>)}</dl></>}
    {supported.length === 0 && <p className="mv-muted">No calibrated photo comparison metric is available for this artifact.</p>}
    {warnings.length > 0 && <><h4>Export warnings</h4><ul className="mv-warning-list">{warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></>}
    <p className="mv-muted">Fit and optical suitability are approximate. This model is not a physical fit or prescription certification.</p>
  </>;
}

function ARView({ project, quality }: { project: Project; quality: Quality }) {
  const modelUrl = project.artifact?.model_url ?? '';
  const [savedFit, setSavedFit] = useState(() => readARFit(modelUrl));
  const stage = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const session = useRef<{ stop: () => void; trackPhoto: (element: HTMLImageElement) => Promise<void> } | null>(null);
  const startup = useRef<AbortController | null>(null);
  const requestId = useRef(0);
  const photoUrl = useRef<string | null>(null);
  const [photo, setPhoto] = useState<string | null>(null);
  const [status, setStatus] = useState('Camera is off');
  const [performanceLabel,setPerformanceLabel]=useState('');
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [denied, setDenied] = useState(false);
  const { pd, scale, horizontal, vertical, depth, headWidth, headHeight, headHorizontal, headVertical, skullDepth } = savedFit;
  const changeFit = (key: keyof typeof defaultARFit, value: number) => setSavedFit(current => ({ ...current, [key]: value }));
  const [showSkull, setShowSkull] = useState(false);
  const [showFaceOccluder, setShowFaceOccluder] = useState(false);
  const [showAnchors, setShowAnchors] = useState(false);
  const [showGlassesWireframe, setShowGlassesWireframe] = useState(false);
  const [softShadow, setSoftShadow] = useState(false);
  const [useFaceOcclusion, setUseFaceOcclusion] = useState(true);
  // A generic skull is not measured to this person. Keep side-head masking
  // opt-in so an oversized shell cannot silently erase the visible temple.
  const [useSkullOcclusion, setUseSkullOcclusion] = useState(false);
  const [previewFit, setPreviewFit] = useState<PreviewFit>('cover');
  const settings = useRef<ARSettings>({ ...savedFit, previewFit, showSkull, showFaceOccluder, showAnchors, showGlassesWireframe, softShadow, useFaceOcclusion, useSkullOcclusion });
  settings.current = { ...savedFit, previewFit, showSkull, showFaceOccluder, showAnchors, showGlassesWireframe, softShadow, useFaceOcclusion, useSkullOcclusion };
  useEffect(() => { saveARFit(modelUrl, savedFit); }, [modelUrl, savedFit]);

  useEffect(() => () => { requestId.current++; startup.current?.abort(); session.current?.stop(); session.current = null; if (photoUrl.current) URL.revokeObjectURL(photoUrl.current); }, []);

  const stop = () => { requestId.current++; startup.current?.abort(); startup.current = null; session.current?.stop(); session.current = null; setRunning(false); setBusy(false); setPerformanceLabel(''); setStatus('Camera is off'); };

  const start = async (usePhoto: boolean) => {
    if (!stage.current || !canvas.current || !modelUrl) return;
    const thisRequest = ++requestId.current;
    startup.current?.abort();
    const controller = new AbortController();
    startup.current = controller;
    session.current?.stop();
    session.current = null;
    setBusy(true);
    setStatus('Loading face tracker and exported model…');
    try {
      const next = await createARSession({ stage: stage.current, canvas: canvas.current, video: video.current, modelUrl, quality, settings, setStatus, setPerformanceLabel, usePhoto, signal: controller.signal });
      if (thisRequest !== requestId.current) { next.stop(); return; }
      startup.current = null;
      session.current = next;
      setBusy(false);
      if (usePhoto) {
        setRunning(false);
        if (image.current?.complete) await next.trackPhoto(image.current);
      } else {
        setRunning(true);
        setDenied(false);
      }
    } catch (error) {
      if (thisRequest !== requestId.current) return;
      startup.current = null;
      session.current?.stop();
      session.current = null;
      setBusy(false);
      setRunning(false);
      if (error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')) {
        setDenied(true);
        setStatus('Camera permission was denied. You can upload a local photo instead.');
      } else setStatus(`AR unavailable: ${errorText(error)}`);
    }
  };

  const useLocalPhoto = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) { setStatus('Choose an image file.'); return; }
    requestId.current++;
    startup.current?.abort();
    startup.current = null;
    session.current?.stop();
    session.current = null;
    if (photoUrl.current) URL.revokeObjectURL(photoUrl.current);
    photoUrl.current = URL.createObjectURL(file);
    setPhoto(photoUrl.current);
    setRunning(false);
    setStatus('Loading local photo…');
  };

  return <div className="mv-workspace">
    <div className="mv-stage-panel">
      <div className="mv-ar-stage" ref={stage}>
        <video ref={video} className="mv-camera" autoPlay playsInline muted style={{ display: running ? 'block' : 'none', objectFit: previewFit }} />
        {photo && <img ref={image} className="mv-camera" style={{ objectFit: previewFit }} src={photo} alt="Local fitting photo" onLoad={() => { void start(true); }} />}
        <canvas ref={canvas} className="mv-ar-overlay" aria-label="Three-dimensional glasses aligned to face" />
        {!running && !photo && <div className="mv-ar-placeholder"><span className="mv-camera-icon">◎</span><h3>Virtual try-on</h3><p>Align the exported glasses with face landmarks in your browser.</p></div>}
        <div className="mv-tracking-status" role="status">{status}{performanceLabel&&<small style={{display:'block'}}>{performanceLabel}</small>}</div>
      </div>
      <div className="mv-ar-actions">
        {running ? <button type="button" className="mv-primary" onClick={stop}>Stop camera</button> : <button type="button" className="mv-primary" disabled={busy} onClick={() => { setPhoto(null); void start(false); }}>{busy ? 'Starting…' : 'Start camera'}</button>}
        <button type="button" disabled={busy} onClick={() => input.current?.click()}>Use a local photo</button>
        <input ref={input} type="file" accept="image/*" hidden onChange={event => useLocalPhoto(event.target.files?.[0])} />
        {denied && <span className="mv-muted">Camera access can also be changed in browser site settings.</span>}
      </div>
      <p className="mv-muted">Face frames and uploaded photos stay in this browser session. No image is sent to the server by this viewer.</p>
    </div>
    <aside className="mv-sidebar">
      <section className="mv-card"><h3>Fit calibration</h3>
        <label className="mv-field">Preview framing <select className="mv-fit-select" value={previewFit} onChange={event => setPreviewFit(event.target.value as PreviewFit)}><option value="cover">Fill / crop</option><option value="contain">Show full frame</option></select></label>
        <label className="mv-field">Pupillary distance <span>{pd} mm</span><input type="range" min="50" max="75" step="1" value={pd} onChange={event => changeFit('pd', Number(event.target.value))} /></label>
        <h4>Glasses fit</h4>
        <label className="mv-field">Frame size <span>{scale}%</span><input type="range" min="70" max="135" step="1" value={scale} onChange={event => changeFit('scale', Number(event.target.value))} /></label>
        <label className="mv-field">Horizontal <span>{horizontal} px</span><input type="range" min="-80" max="80" value={horizontal} onChange={event => changeFit('horizontal', Number(event.target.value))} /></label>
        <label className="mv-field">Vertical <span>{vertical} px</span><input type="range" min="-80" max="80" value={vertical} onChange={event => changeFit('vertical', Number(event.target.value))} /></label>
        <label className="mv-field">Depth <span>{depth} mm</span><input type="range" min="-30" max="30" value={depth} onChange={event => changeFit('depth', Number(event.target.value))} /></label>
        <h4>Head fit</h4>
        <label className="mv-field">Head width <span>{headWidth}%</span><input type="range" min="80" max="120" value={headWidth} onChange={event => changeFit('headWidth', Number(event.target.value))} /></label>
        <label className="mv-field">Head height <span>{headHeight}%</span><input type="range" min="80" max="120" value={headHeight} onChange={event => changeFit('headHeight', Number(event.target.value))} /></label>
        <label className="mv-field">Head horizontal <span>{headHorizontal} mm</span><input type="range" min="-20" max="20" value={headHorizontal} onChange={event => changeFit('headHorizontal', Number(event.target.value))} /></label>
        <label className="mv-field">Head vertical <span>{headVertical} mm</span><input type="range" min="-20" max="20" value={headVertical} onChange={event => changeFit('headVertical', Number(event.target.value))} /></label>
        <label className="mv-field">Skull depth <span>{(skullDepth / 10).toFixed(1)} cm</span><input type="range" min="-30" max="30" step="1" value={skullDepth} onChange={event => changeFit('skullDepth', Number(event.target.value))} /></label>
        <button type="button" onClick={() => setSavedFit({ ...defaultARFit })}>Reset calibration</button>
      </section>
      <section className="mv-card"><h3>Occlusion diagnostics</h3><div className="mv-ar-toggles">
        <label><input type="checkbox" checked={useFaceOcclusion} onChange={event => setUseFaceOcclusion(event.target.checked)} /> Use live face depth mask</label>
        <label><input type="checkbox" checked={useSkullOcclusion} onChange={event => setUseSkullOcclusion(event.target.checked)} /> Use skull side-head mask (calibrate first)</label>
        <p className="mv-muted">Live face masking is on by default. The generic skull is optional: an oversized or misaligned skull can hide the near arm. Show its alignment and adjust Head width / Skull depth before enabling it. To check whether an arm exists, temporarily turn both depth masks off; far arms will then show through the head.</p>
        <label><input type="checkbox" checked={showSkull} onChange={event => setShowSkull(event.target.checked)} /> Show skull alignment</label>
        <label><input type="checkbox" checked={showFaceOccluder} onChange={event => setShowFaceOccluder(event.target.checked)} /> Show live face occluder</label>
        <label><input type="checkbox" checked={showAnchors} onChange={event => setShowAnchors(event.target.checked)} /> Show fitted/detected anchors</label>
        <label><input type="checkbox" checked={showGlassesWireframe} onChange={event => setShowGlassesWireframe(event.target.checked)} /> Show glasses wireframe</label>
        <label><input type="checkbox" checked={softShadow} onChange={event => setSoftShadow(event.target.checked)} /> Soft frame shadow</label>
      </div><p className="mv-muted">Cyan marks the fitted head anchors; magenta marks detected face landmarks. The skull wireframe is a generic alignment guide.</p></section>
      <section className="mv-card"><h3>About this preview</h3><p className="mv-muted">PD sets the approximate on-screen scale. Enter your measured PD if known, then adjust placement manually. Camera perspective and face shape can make the preview differ from a physical fit.</p><p className="mv-muted">The tracked face mesh and approximate skull hide rendered glasses where they pass behind the face. Depth rendering cannot remove glasses already present in a camera image. This is not fit, safety, or prescription certification.</p></section>
    </aside>
  </div>;
}

async function createARSession(options: {
  stage: HTMLDivElement;
  canvas: HTMLCanvasElement;
  video: HTMLVideoElement | null;
  modelUrl: string;
  quality: Quality;
  settings: MutableRefObject<ARSettings>;
  setStatus: (status: string) => void;
  setPerformanceLabel:(label:string)=>void;
  usePhoto: boolean;
  signal?: AbortSignal;
}) {
  const { stage, canvas, video, modelUrl, quality, settings, setStatus, setPerformanceLabel, usePhoto, signal } = options;
  let disposed = false;
  let stream: MediaStream | null = null;
  let frame = 0;
  let lastVideoTime = -1;
  let lastDetection = 0;
  let tracker: Awaited<ReturnType<typeof import('@mediapipe/tasks-vision').FaceLandmarker.createFromOptions>> | null = null;
  let model: THREE.Object3D | null = null;
  const renderer = makeRenderer(canvas, quality, true);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 2000);
  camera.position.z = 1000;
  const pose = new THREE.Group();
  const glassesFit = new THREE.Group();
  const headFit = new THREE.Group();
  const skullOffset = new THREE.Group();
  const anchorDebug = new THREE.Group();
  pose.add(headFit, glassesFit, anchorDebug);
  headFit.add(skullOffset);
  scene.add(pose);
  scene.add(new THREE.HemisphereLight('#ffffff', '#9bb7ae', 2.5));
  const light = new THREE.DirectionalLight('#ffffff', 2.7);
  light.position.set(0.2, 0.4, 1);
  scene.add(light);
  let skull: ReturnType<typeof createSkullOccluder> | null = null;
  let face: ReturnType<typeof createFaceOccluder> | null = null;
  let faceInitialized = false;
  let appliedWireframe: boolean | null = null;
  const fittedMarkers: THREE.Mesh[] = [];
  const detectedMarkers: THREE.Mesh[] = [];
  for (const index of FACE_ANCHORS) {
    const geometry = new THREE.SphereGeometry(.0016, 8, 6);
    const fitted = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x00e5ff, depthTest: false, depthWrite: false }));
    const detected = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xff4fd8, depthTest: false, depthWrite: false }));
    fitted.renderOrder = detected.renderOrder = 200;
    anchorDebug.add(fitted, detected);
    fittedMarkers.push(fitted);
    detectedMarkers.push(detected);
  }
  anchorDebug.visible = false;
  const resize = () => {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.left = -width / 2;
    camera.right = width / 2;
    camera.top = height / 2;
    camera.bottom = -height / 2;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(stage);
  resize();
  const stop = () => {
    if (disposed) return;
    disposed = true;
    signal?.removeEventListener('abort', stop);
    cancelAnimationFrame(frame);
    observer.disconnect();
    stream?.getTracks().forEach(track => track.stop());
    if (video) { video.pause(); video.srcObject = null; }
    tracker?.close();
    if (model) disposeObject(model);
    if (skull) {
      // Debug and shadow share the depth geometry; release it once.
      skull.depth.traverse(node => { if (node instanceof THREE.Mesh) { node.geometry.dispose(); (node.material as THREE.Material).dispose(); } });
      skull.debug.traverse(node => { if (node instanceof THREE.Mesh) (node.material as THREE.Material).dispose(); });
      skull.shadow.traverse(node => { if (node instanceof THREE.Mesh) (node.material as THREE.Material).dispose(); });
    }
    if (face) { face.geometry.dispose(); (face.depth.material as THREE.Material).dispose(); (face.debug.material as THREE.Material).dispose(); (face.shadow.material as THREE.Material).dispose(); }
    fittedMarkers.forEach(marker => { marker.geometry.dispose(); (marker.material as THREE.Material).dispose(); });
    detectedMarkers.forEach(marker => (marker.material as THREE.Material).dispose());
    renderer.dispose();
    // The same canvas is reused when switching camera/photo sessions. Forcing
    // context loss here prevents the next renderer from initializing reliably.
  };
  signal?.addEventListener('abort', stop, { once: true });
  if (signal?.aborted) stop();

  try {
    const assets = await Promise.allSettled([
      new GLTFLoader().loadAsync(modelUrl),
      new GLTFLoader().loadAsync('/models/head-occluder.glb'),
      fetch('/models/canonical_face_model.obj', { signal }),
      import('@mediapipe/tasks-vision'),
    ] as const);
    if (disposed || assets.some(asset => asset.status === 'rejected')) {
      if (assets[0].status === 'fulfilled') disposeObject(assets[0].value.scene);
      if (assets[1].status === 'fulfilled') disposeObject(assets[1].value.scene);
      const failure = assets.find(asset => asset.status === 'rejected');
      throw failure?.status === 'rejected' ? failure.reason : new Error('Session stopped');
    }
    if (assets[0].status !== 'fulfilled' || assets[1].status !== 'fulfilled' || assets[2].status !== 'fulfilled' || assets[3].status !== 'fulfilled') throw new Error('Could not load AR assets');
    const gltf = assets[0].value;
    const skullGltf = assets[1].value;
    const topologyResponse = assets[2].value;
    const vision = assets[3].value;
    if (!topologyResponse.ok) { disposeObject(gltf.scene); disposeObject(skullGltf.scene); throw new Error('Could not load face mesh topology'); }
    let topologySource: string;
    try { topologySource = await topologyResponse.text(); }
    catch (error) { disposeObject(gltf.scene); disposeObject(skullGltf.scene); throw error; }
    if (disposed) { disposeObject(gltf.scene); disposeObject(skullGltf.scene); throw new Error('Session stopped'); }
    model = gltf.scene;
    preservePhotoTextureColors(model);
    preserveTextureDetail(model, renderer);
    glassesFit.add(model);
    try { skull = createSkullOccluder(skullGltf.scene); }
    finally { disposeObject(skullGltf.scene); }
    skullOffset.add(skull.root);
    const topology = parseFaceTopology(topologySource);
    face = createFaceOccluder(topology);
    pose.add(face.depth, face.shadow, face.debug);
    FACE_ANCHORS.forEach((index, i) => fittedMarkers[i].position.copy(topology.vertices[index]).sub(CANONICAL_BRIDGE).multiplyScalar(.01));
    const frameLight = new THREE.DirectionalLight('#ffffff', .4);
    frameLight.position.set(-300, 400, 500);
    frameLight.castShadow = true;
    frameLight.shadow.mapSize.set(512, 512);
    frameLight.shadow.camera.left = -400; frameLight.shadow.camera.right = 400;
    frameLight.shadow.camera.top = 400; frameLight.shadow.camera.bottom = -400;
    frameLight.shadow.camera.near = 1; frameLight.shadow.camera.far = 1600;
    frameLight.target.position.set(0, 0, 0);
    scene.add(frameLight, frameLight.target);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    configureFrameShadows(model);
    const fileset = await vision.FilesetResolver.forVisionTasks('/wasm');
    if (disposed) throw new Error('Session stopped');
    try {
      tracker = await vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'GPU' },
        runningMode: usePhoto ? 'IMAGE' : 'VIDEO', numFaces: 1,
        minFaceDetectionConfidence: 0.55, minFacePresenceConfidence: 0.55, minTrackingConfidence: 0.55,
        outputFacialTransformationMatrixes: true,
      });
    } catch {
      tracker = await vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
        runningMode: usePhoto ? 'IMAGE' : 'VIDEO', numFaces: 1,
        outputFacialTransformationMatrixes: true,
      });
    }
    if (disposed) { tracker?.close(); tracker = null; throw new Error('Session stopped'); }
    if (!usePhoto) {
      if (!video || !navigator.mediaDevices?.getUserMedia) throw new Error('Camera is unavailable in this browser or context');
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: quality === 'low' ? 640 : 1280 }, height: { ideal: quality === 'low' ? 480 : 960 } }, audio: false });
      if (disposed) { stream.getTracks().forEach(track => track.stop()); throw new Error('Session stopped'); }
      video.srcObject = stream;
      await video.play();
      setStatus('Looking for a face…');
    }
    const smooth = { initialized: false, x: 0, y: 0, pxPerM: 0, quaternion: new THREE.Quaternion() };
    const faceTransform = new THREE.Matrix4();
    const faceRotation = new THREE.Matrix4();
    const faceEuler = new THREE.Euler();
    const targetQuaternion = new THREE.Quaternion();
    const updatePose = (landmarks: Landmark[] | undefined, matrix: FaceMatrix | undefined, sourceWidth: number, sourceHeight: number) => {
      if (!landmarks || !landmarks[168] || !landmarks[33] || !landmarks[263]) { pose.visible = false; smooth.initialized = false; faceInitialized = false; return false; }
      const rect = stage.getBoundingClientRect();
      // Match CSS object-fit exactly so landmarks stay registered through crop or letterbox.
      const fittedScale = settings.current.previewFit === 'cover'
        ? Math.max(rect.width / sourceWidth, rect.height / sourceHeight)
        : Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
      const displayedWidth = sourceWidth * fittedScale;
      const displayedHeight = sourceHeight * fittedScale;
      const offsetX = (rect.width - displayedWidth) / 2;
      const offsetY = (rect.height - displayedHeight) / 2;
      const map = (point: Landmark) => ({ x: (1 - point.x) * displayedWidth + offsetX, y: point.y * displayedHeight + offsetY });
      const bridge = map(landmarks[168]);
      const left = map(landmarks[468] ?? landmarks[33]);
      const right = map(landmarks[473] ?? landmarks[263]);
      const eyeDistance = Math.hypot(left.x - right.x, left.y - right.y);
      if (!Number.isFinite(eyeDistance) || eyeDistance < 8) { pose.visible = false; smooth.initialized = false; faceInitialized = false; return false; }
      const calibration = settings.current;
      const x = bridge.x - rect.width / 2;
      const y = rect.height / 2 - bridge.y;
      let yawForScale=0;
      if (matrix?.data?.length === 16) {
        faceTransform.fromArray(matrix.data);
        faceRotation.extractRotation(faceTransform);
        faceEuler.setFromRotationMatrix(faceRotation, 'YXZ');
        yawForScale=faceEuler.y;
        faceEuler.set(clamp(faceEuler.x, -0.6, 0.6), clamp(-faceEuler.y, -0.85, 0.85), clamp(-faceEuler.z, -0.7, 0.7), 'YXZ');
      } else {
        faceEuler.set(0, 0, -Math.atan2(right.y - left.y, right.x - left.x), 'YXZ');
      }
      const pxPerM=pixelsPerMeter(eyeDistance,calibration.pd,yawForScale);
      targetQuaternion.setFromEuler(faceEuler);
      if (!smooth.initialized) {
        smooth.x = x; smooth.y = y; smooth.pxPerM = pxPerM; smooth.quaternion.copy(targetQuaternion); smooth.initialized = true;
      } else {
        smooth.x += (x - smooth.x) * 0.32;
        smooth.y += (y - smooth.y) * 0.32;
        smooth.pxPerM += (pxPerM - smooth.pxPerM) * 0.28;
        smooth.quaternion.slerp(targetQuaternion, 0.28);
      }
      pose.position.set(smooth.x, smooth.y, 0);
      frameLight.position.set(smooth.x - 300, smooth.y + 400, 500);
      frameLight.target.position.set(smooth.x, smooth.y, 0);
      frameLight.target.updateMatrixWorld();
      pose.quaternion.copy(smooth.quaternion);
      // Video and image previews are mirrored; +X in the GLB is the wearer's left.
      pose.scale.set(-smooth.pxPerM, smooth.pxPerM, smooth.pxPerM);
      glassesFit.position.set(-calibration.horizontal / smooth.pxPerM, -calibration.vertical / smooth.pxPerM, calibration.depth / 1000);
      glassesFit.scale.setScalar(calibration.scale / 100);
      headFit.position.set(calibration.headHorizontal / 1000, calibration.headVertical / 1000, 0);
      headFit.scale.set(calibration.headWidth / 100, calibration.headHeight / 100, (calibration.headWidth + calibration.headHeight) / 200);
      skullOffset.position.z = calibration.skullDepth / 1000;
      applyOcclusionVisibility(face, skull, calibration);
      anchorDebug.visible = calibration.showAnchors;
      frameLight.castShadow = calibration.softShadow;
      if (appliedWireframe !== calibration.showGlassesWireframe) {
        model?.traverse(node => { if (node instanceof THREE.Mesh) (Array.isArray(node.material) ? node.material : [node.material]).forEach(material => { if ('wireframe' in material) (material as THREE.MeshStandardMaterial).wireframe = calibration.showGlassesWireframe; }); });
        appliedWireframe = calibration.showGlassesWireframe;
      }
      if (face) {
        updateFaceOccluder(face.geometry, landmarks, sourceWidth, sourceHeight, fittedScale, smooth.pxPerM, smooth.quaternion, faceInitialized ? .55 : 1);
        faceInitialized = true;
        for (let i = 0; i < FACE_ANCHORS.length; i++) {
          const index = FACE_ANCHORS[i];
          const point = face.geometry.getAttribute('position');
          const canonical = topology.vertices[index];
          fittedMarkers[i].position.set((canonical.x - CANONICAL_BRIDGE.x) * .01 * headFit.scale.x + headFit.position.x, (canonical.y - CANONICAL_BRIDGE.y) * .01 * headFit.scale.y + headFit.position.y, (canonical.z - CANONICAL_BRIDGE.z) * .01 * headFit.scale.z + skullOffset.position.z);
          detectedMarkers[i].position.set(point.getX(index), point.getY(index), point.getZ(index) + .003);
        }
      }
      pose.visible = true;
      return true;
    };
    let stillFace: { landmarks: Landmark[] | undefined; matrix: FaceMatrix | undefined; width: number; height: number } | null = null;
    let performanceAt=performance.now(),renderCount=0,trackingCount=0;
    const render = () => {
      if (disposed) return;
      frame = requestAnimationFrame(render);
      if (!usePhoto && video && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        const now = performance.now();
        const interval = quality === 'low' ? 66 : quality === 'balanced' ? 40 : 0;
        if (now - lastDetection >= interval) {
          lastDetection = now;
          lastVideoTime = video.currentTime;
          try {
            const result = tracker!.detectForVideo(video, now);
            trackingCount++;
            setStatus(updatePose(result.faceLandmarks[0], result.facialTransformationMatrixes?.[0], video.videoWidth, video.videoHeight) ? 'Face tracked · approximate placement' : 'Move your face into view');
          } catch (error) { pose.visible = false; faceInitialized = false; setStatus(`Tracking paused: ${errorText(error)}`); }
        }
      }
      if (usePhoto && stillFace) updatePose(stillFace.landmarks, stillFace.matrix, stillFace.width, stillFace.height);
      renderer.render(scene, camera);
      renderCount++;
      const elapsed=performance.now()-performanceAt;
      if(elapsed>=1000){setPerformanceLabel(`${Math.round(renderCount*1000/elapsed)} render FPS · ${Math.round(trackingCount*1000/elapsed)} tracking updates/s`);renderCount=0;trackingCount=0;performanceAt=performance.now();}
    };
    render();
    return {
      stop,
      async trackPhoto(element: HTMLImageElement) {
        if (disposed || !tracker || !element.naturalWidth) return;
        const result = tracker.detect(element);
        stillFace = { landmarks: result.faceLandmarks[0], matrix: result.facialTransformationMatrixes?.[0], width: element.naturalWidth, height: element.naturalHeight };
        setStatus(updatePose(stillFace.landmarks, stillFace.matrix, stillFace.width, stillFace.height) ? 'Face aligned · approximate placement' : 'No face found in this photo');
      },
    };
  } catch (error) { stop(); throw error; }
}

export default function ModelViewer({ project, mode = 'inspect', onModeChange }: ViewerProps) {
  const [internalMode, setInternalMode] = useState<'inspect' | 'ar'>(mode);
  const [quality, setQuality] = useState<Quality>('balanced');
  const activeMode = onModeChange ? mode : internalMode;
  useEffect(() => { setInternalMode(mode); }, [mode]);
  const artifact = project.artifact;
  const canUse = Boolean(artifact?.model_url);
  const current = useMemo(() => !artifact || artifact.revision !== project.revision || project.stale, [artifact, project.revision, project.stale]);
  const changeMode = (next: 'inspect' | 'ar') => { setInternalMode(next); onModeChange?.(next); };
  return <section className="mv-root" aria-label="3D model viewer">
    <div className="mv-header"><div><span className="mv-kicker">MODEL REVIEW</span><h2>{project.name}</h2><p>Inspect the export and preview approximate placement on a face.</p></div><div className="mv-header-controls"><label>Quality <select value={quality} onChange={event => setQuality(event.target.value as Quality)}><option value="low">Low</option><option value="balanced">Balanced</option><option value="high">High</option></select></label><div className="mv-mode-switch"><button type="button" className={activeMode === 'inspect' ? 'active' : ''} onClick={() => changeMode('inspect')}>3D inspect</button><button type="button" className={activeMode === 'ar' ? 'active' : ''} onClick={() => changeMode('ar')}>Live AR</button></div></div></div>
    {current && <div className="mv-warning">{artifact ? 'This artifact belongs to an earlier revision. Regenerate to review current edits.' : 'Generate a model to enable 3D inspection and AR.'}</div>}
    {canUse ? activeMode === 'inspect' ? <Inspection project={project} quality={quality} /> : <ARView key={`${artifact!.model_url}-${quality}`} project={project} quality={quality} /> : <div className="mv-empty">The exported GLB will appear here after generation.</div>}
  </section>;
}
