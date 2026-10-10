import * as THREE from 'three';

// Shadow depth maps do not reproduce optical lens transmission. Let only the
// structural parts cast the optional frame shadow, never an opaque eye patch.
export function configureFrameShadows(root: THREE.Object3D) {
  root.traverse(node => {
    if (node instanceof THREE.Mesh) node.castShadow = !/^lens(?:_|-|$)/i.test(node.name);
  });
}

export type FacePoint = { x: number; y: number; z: number };

export function applyOcclusionVisibility(
  face: { depth: THREE.Object3D; shadow: THREE.Object3D; debug: THREE.Object3D } | null,
  skull: { depth: THREE.Object3D; shadow: THREE.Object3D; debug: THREE.Object3D } | null,
  settings: { useFaceOcclusion: boolean; useSkullOcclusion: boolean; softShadow: boolean; showFaceOccluder: boolean; showSkull: boolean },
) {
  if (face) {
    face.depth.visible = settings.useFaceOcclusion;
    face.shadow.visible = settings.useFaceOcclusion && settings.softShadow;
    face.debug.visible = settings.showFaceOccluder;
  }
  if (skull) {
    skull.depth.visible = settings.useSkullOcclusion;
    skull.shadow.visible = settings.useSkullOcclusion && settings.softShadow;
    skull.debug.visible = settings.showSkull;
  }
}

export const FACE_ANCHORS = [1, 10, 33, 133, 168, 234, 263, 362, 454] as const;
export const CANONICAL_BRIDGE = new THREE.Vector3(0, 3.271027, 5.236015);
const CANONICAL_NOSE = new THREE.Vector3(0, -1.126865, 7.475604);
const CANONICAL_CHIN = new THREE.Vector3(0, -9.403378, 4.264492);
const CANONICAL_WIDTH = 15.328364;

export function parseFaceTopology(source: string) {
  const vertices: THREE.Vector3[] = [];
  const indices: number[] = [];
  for (const line of source.split(/\r?\n/)) {
    if (line.startsWith('v ')) {
      const values = line.trim().split(/\s+/).slice(1, 4).map(Number);
      vertices.push(new THREE.Vector3(values[0], values[1], values[2]));
    } else if (line.startsWith('f ')) {
      const face = line.trim().split(/\s+/).slice(1).map(token => Number(token.split('/')[0]) - 1);
      for (let i = 1; i + 1 < face.length; i++) indices.push(face[0], face[i], face[i + 1]);
    }
  }
  if (vertices.length < 468 || !indices.length || indices.some(index => index < 0 || index >= vertices.length)) throw new Error('Invalid canonical face topology');
  return { vertices, indices };
}

function averageVertices(source: THREE.Object3D, predicate: (point: THREE.Vector3) => boolean) {
  const sum = new THREE.Vector3();
  const point = new THREE.Vector3();
  let count = 0;
  source.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return;
    const positions = node.geometry.getAttribute('position');
    if (!positions) return;
    for (let i = 0; i < positions.count; i++) {
      point.fromBufferAttribute(positions, i).applyMatrix4(node.matrixWorld);
      if (predicate(point)) { sum.add(point); count++; }
    }
  });
  return count ? sum.multiplyScalar(1 / count) : null;
}

function clipSkullFront(material: THREE.Material, center: THREE.Vector3, size: THREE.Vector3) {
  material.onBeforeCompile = shader => {
    shader.uniforms.skullCenter = { value: center.clone() };
    shader.uniforms.skullSize = { value: size.clone() };
    shader.vertexShader = `varying vec3 vSkullPoint;\n${shader.vertexShader}`.replace('#include <begin_vertex>', '#include <begin_vertex>\nvSkullPoint = position;');
    shader.fragmentShader = `varying vec3 vSkullPoint; uniform vec3 skullCenter; uniform vec3 skullSize;\n${shader.fragmentShader}`.replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
      float side = abs(vSkullPoint.x - skullCenter.x) / max(skullSize.x * 0.5, 0.0001);
      float front = (vSkullPoint.z - (skullCenter.z - skullSize.z * 0.5)) / max(skullSize.z, 0.0001);
      if (side < 0.68 && front > 0.46) discard;
    `);
  };
  material.customProgramCacheKey = () => 'side-back-skull-v1';
}

export function createSkullOccluder(source: THREE.Object3D) {
  source.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(source);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const nose = averageVertices(source, point => point.z > bounds.max.z - size.z * .03 && Math.abs(point.x - center.x) < size.x * .15);
  const chin = averageVertices(source, point => point.y < bounds.min.y + size.y * .05 && Math.abs(point.x - center.x) < size.x * .2);
  if (!nose || !chin) throw new Error('Could not register skull nose and chin');
  const sx = CANONICAL_WIDTH / Math.max(size.x, .001) / 100;
  const sy = (CANONICAL_NOSE.y - CANONICAL_CHIN.y) / Math.max(nose.y - chin.y, .001) / 100;
  const root = new THREE.Group();
  root.scale.set(sx, sy, sy);
  root.position.set(-nose.x * sx, (CANONICAL_NOSE.y - CANONICAL_BRIDGE.y) / 100 - nose.y * sy, (CANONICAL_NOSE.z - CANONICAL_BRIDGE.z) / 100 - nose.z * sy);
  const depth = new THREE.Group();
  const debug = new THREE.Group();
  const shadow = new THREE.Group();
  source.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return;
    const geometry = node.geometry.clone();
    geometry.applyMatrix4(node.matrixWorld);
    const attr = geometry.getAttribute('position');
    const point = new THREE.Vector3();
    for (let i = 0; i < attr.count; i++) {
      point.fromBufferAttribute(attr, i);
      const side = Math.abs(point.x - center.x) / Math.max(size.x / 2, .001);
      const band = 1 - THREE.MathUtils.smoothstep(Math.abs(point.y - nose.y), size.y * .08, size.y * .3);
      const inset = .11 + .14 * band * THREE.MathUtils.smoothstep(side, .3, .85);
      attr.setXYZ(i, THREE.MathUtils.lerp(point.x, center.x, inset), THREE.MathUtils.lerp(point.y, center.y, .11), THREE.MathUtils.lerp(point.z, center.z, .11));
    }
    attr.needsUpdate = true;
    const depthMaterial = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, depthTest: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    // The live mesh covers the face; the generic skull contributes the sides and back.
    clipSkullFront(depthMaterial, center, size);
    const depthMesh = new THREE.Mesh(geometry, depthMaterial);
    depthMesh.frustumCulled = false;
    depthMesh.renderOrder = -120;
    depth.add(depthMesh);
    const debugMesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x70f0bc, wireframe: true, transparent: true, opacity: .5, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
    debugMesh.renderOrder = 100;
    debug.add(debugMesh);
    const shadowMaterial = new THREE.ShadowMaterial({ color: 0x000000, opacity: .2, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    clipSkullFront(shadowMaterial, center, size);
    const shadowMesh = new THREE.Mesh(geometry, shadowMaterial);
    shadowMesh.receiveShadow = true;
    shadowMesh.renderOrder = -110;
    shadow.add(shadowMesh);
  });
  root.add(depth, shadow, debug);
  debug.visible = false;
  shadow.visible = false;
  return { root, depth, debug, shadow };
}

export function createFaceOccluder(topology: ReturnType<typeof parseFaceTopology>) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(topology.vertices.length * 3), 3));
  geometry.setIndex(topology.indices);
  const depth = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
  depth.renderOrder = -130;
  depth.frustumCulled = false;
  const debug = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xff4fd8, wireframe: true, transparent: true, opacity: .55, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
  debug.renderOrder = 110;
  debug.frustumCulled = false;
  debug.visible = false;
  const shadow = new THREE.Mesh(geometry, new THREE.ShadowMaterial({ color: 0x000000, opacity: .2, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  shadow.receiveShadow = true;
  shadow.renderOrder = -115;
  shadow.frustumCulled = false;
  shadow.visible = false;
  return { depth, debug, shadow, geometry };
}

export function updateFaceOccluder(geometry: THREE.BufferGeometry, landmarks: FacePoint[], sourceWidth: number, sourceHeight: number, fittedScale: number, pixelsPerM: number, rotation: THREE.Quaternion, blend = 1) {
  const positions = geometry.getAttribute('position') as THREE.BufferAttribute;
  const bridge = landmarks[168];
  if (!bridge || !pixelsPerM) return false;
  const inverse = rotation.clone().invert();
  const point = new THREE.Vector3();
  const count = Math.min(positions.count, landmarks.length);
  for (let i = 0; i < count; i++) {
    const landmark = landmarks[i];
    if (!landmark) continue;
    const dx = -(landmark.x - bridge.x) * sourceWidth * fittedScale;
    const dy = -(landmark.y - bridge.y) * sourceHeight * fittedScale;
    const dz = -(landmark.z - bridge.z) * sourceWidth * fittedScale;
    point.set(dx, dy, dz).applyQuaternion(inverse);
    // The parent has a negative X scale for the mirrored preview.
    point.set(-point.x / pixelsPerM, point.y / pixelsPerM, point.z / pixelsPerM);
    // Pull the depth surface slightly into the face to preserve rims at the bridge.
    point.z -= .002 + (Math.abs(point.x) < .032 ? .005 : 0);
    positions.setXYZ(i, THREE.MathUtils.lerp(positions.getX(i), point.x, blend), THREE.MathUtils.lerp(positions.getY(i), point.y, blend), THREE.MathUtils.lerp(positions.getZ(i), point.z, blend));
  }
  positions.needsUpdate = true;
  return true;
}
