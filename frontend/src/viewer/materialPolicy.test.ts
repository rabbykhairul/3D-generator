import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { preservePhotoTextureColors } from './materialPolicy';

function check(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function sameMatrix(actual: THREE.Matrix4, expected: THREE.Matrix4, label: string) {
  check(actual.elements.every((value, index) => Math.abs(value - expected.elements[index]) < 1e-6), label);
}

function mesh(root: THREE.Object3D, name: string) {
  const part = root.getObjectByName(name);
  check(part instanceof THREE.Mesh, `${name} exists as a mesh`);
  return part;
}

function singleMaterial(part: THREE.Mesh) {
  check(!Array.isArray(part.material), `${part.name} has one material`);
  return part.material;
}

export async function testPhotoMaterialPolicy(modelUrl: string) {
  const model = (await new GLTFLoader().loadAsync(modelUrl)).scene;
  const names = ['front_frame', 'lens_left', 'lens_right', 'hinge_left', 'temple_left', 'hinge_right', 'temple_right'];
  const parts = names.map(name => mesh(model, name));
  const originals = parts.map(part => ({
    parent: part.parent,
    local: part.matrix.clone(),
    material: singleMaterial(part),
  }));
  const front = originals[0].material;
  const temple = originals[4].material;
  check(front instanceof THREE.MeshStandardMaterial && Boolean(front.map), 'front has a photo texture');
  check(temple instanceof THREE.MeshStandardMaterial && Boolean(temple.map), 'temple has a photo texture');
  check(front.map !== temple.map, 'front and temple retain independent photo textures');
  for (const index of [1, 2]) {
    const material = originals[index].material;
    check(material instanceof THREE.MeshStandardMaterial && !material.map, `${names[index]} remains an untextured lens`);
  }

  preservePhotoTextureColors(model);
  for (const [index, part] of parts.entries()) {
    check(part.parent === originals[index].parent, `${part.name} parent is unchanged`);
    sameMatrix(part.matrix, originals[index].local, `${part.name} local transform is unchanged`);
  }
  const texturedParts: string[] = [];
  for (const [index, part] of parts.entries()) {
    const before = originals[index].material;
    if (!(before instanceof THREE.MeshStandardMaterial) || !before.map) continue;
    texturedParts.push(names[index]);
    const after = singleMaterial(part);
    check(after instanceof THREE.MeshBasicMaterial, `${names[index]} uses the shared photo-color policy`);
    check(after.map === before.map, `${names[index]} keeps its own texture`);
    check(after.alphaMap === before.alphaMap && after.vertexColors === before.vertexColors, `${names[index]} keeps alpha map and vertex colors`);
    check(after.color.equals(before.color), `${names[index]} keeps its color factor`);
    check(after.opacity === before.opacity && after.transparent === before.transparent, `${names[index]} keeps alpha`);
    check(after.depthTest === before.depthTest && after.premultipliedAlpha === before.premultipliedAlpha, `${names[index]} keeps depth and alpha handling`);
    check(after.toneMapped === false, `${names[index]} bypasses tone mapping`);
  }
  for (const [index, part] of parts.entries()) {
    const before = originals[index].material;
    if (!(before instanceof THREE.MeshStandardMaterial) || before.map) continue;
    const after = singleMaterial(parts[index]);
    if (names[index].startsWith('hinge_')) {
      check(after instanceof THREE.MeshBasicMaterial && !after.toneMapped, `${names[index]} hardware bypasses relighting`);
    } else {
      check(after === before && after instanceof THREE.MeshStandardMaterial, `${names[index]} authored optical material stays lit`);
    }
    check(after.color.equals(before.color) && after.opacity === before.opacity, `${names[index]} tint and opacity stay unchanged`);
  }

  // Synthetic colored products cover GLBs whose base-color factors are not white,
  // plus mapped alpha/vertex colors and untextured metal hardware.
  const redMap = new THREE.DataTexture(new Uint8Array([190, 25, 30, 255]), 1, 1);
  const greenMap = new THREE.DataTexture(new Uint8Array([20, 120, 35, 255]), 1, 1);
  const alphaMap = new THREE.DataTexture(new Uint8Array([200, 200, 200, 255]), 1, 1);
  const coloredRoot = new THREE.Group();
  const red = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ map: redMap, color: '#a94433', alphaMap, vertexColors: true, transparent: true, opacity: 0.7, premultipliedAlpha: true }));
  const green = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ map: greenMap, color: '#327b56' }));
  const hardware = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ color: '#46494d', metalness: 0.8 }));
  coloredRoot.add(red, green, hardware);
  const redBefore = red.material as THREE.MeshStandardMaterial;
  const greenBefore = green.material as THREE.MeshStandardMaterial;
  const hardwareBefore = hardware.material;
  preservePhotoTextureColors(coloredRoot);
  check(red.material instanceof THREE.MeshBasicMaterial && green.material instanceof THREE.MeshBasicMaterial, 'two synthetic colored textures become unlit');
  check(red.material.map === redMap && green.material.map === greenMap && red.material.color.equals(redBefore.color) && green.material.color.equals(greenBefore.color), 'distinct color maps and factors survive');
  check(red.material.alphaMap === alphaMap && red.material.vertexColors && red.material.premultipliedAlpha && red.material.opacity === 0.7, 'synthetic alpha and vertex colors survive');
  check(hardware.material instanceof THREE.MeshBasicMaterial && !hardware.material.toneMapped, 'untextured hardware bypasses relighting');
  check(hardware.material.color.equals((hardwareBefore as THREE.MeshStandardMaterial).color), 'hardware linear color factor is preserved');

  // AR places the whole GLB under one mirrored, scaled pose. Rotating that pose
  // must preserve every part's transform relative to the model root.
  model.updateMatrixWorld(true);
  const originalRelative = parts.map(part => new THREE.Matrix4().copy(model.matrixWorld).invert().multiply(part.matrixWorld));
  const pose = new THREE.Group();
  pose.add(model);
  for (const [yaw, pitch, roll] of [[0, 0, 0], [0.45, 0.15, -0.2], [-0.7, -0.25, 0.35]]) {
    pose.position.set(47, -23, 6);
    pose.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));
    pose.scale.set(-1400, 1400, 1400);
    pose.updateMatrixWorld(true);
    for (const [index, part] of parts.entries()) {
      const relative = new THREE.Matrix4().copy(model.matrixWorld).invert().multiply(part.matrixWorld);
      sameMatrix(relative, originalRelative[index], `${part.name} stays attached at yaw ${yaw}`);
    }
  }
  return { parts: names, texturedParts, poses: 3 };
}
