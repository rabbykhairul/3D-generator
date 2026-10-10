import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { applyOcclusionVisibility, configureFrameShadows, createFaceOccluder, createSkullOccluder, parseFaceTopology, updateFaceOccluder } from './occlusion.ts';

const glasses = new THREE.Group();
for(const name of ['front_frame','temple_left','hinge_right','lens_left','lens_right']) {
  const part = new THREE.Mesh(); part.name = name; glasses.add(part);
}
configureFrameShadows(glasses);
for(const part of glasses.children)assert.equal(part.castShadow,!part.name.startsWith('lens_'),'clear/tinted lenses must not cast opaque eye patches');

const source = readFileSync(new URL('../../public/models/canonical_face_model.obj', import.meta.url), 'utf8');
const topology = parseFaceTopology(source);
assert.ok(topology.vertices.length >= 468);
assert.ok(topology.indices.length > 1000);
const face = createFaceOccluder(topology);
assert.equal(face.depth.material.colorWrite, false);
assert.equal(face.depth.material.depthWrite, true);
assert.equal(face.debug.material.depthWrite, false);
assert.equal(face.debug.visible, false);
assert.equal(face.shadow.geometry, face.geometry, 'live shadow follows the tracked face vertices');
assert.equal(face.shadow.receiveShadow, true);
assert.equal(face.shadow.material.depthWrite, false);
assert.equal(face.shadow.visible, false);

const landmarks = Array.from({ length: topology.vertices.length }, () => ({ x: .5, y: .5, z: 0 }));
landmarks[168] = { x: .5, y: .5, z: 0 };
landmarks[33] = { x: .4, y: .45, z: -.01 };
assert.equal(updateFaceOccluder(face.geometry, landmarks, 1000, 500, 1, 1000, new THREE.Quaternion()), true);
const positions = face.geometry.getAttribute('position');
assert.ok(Math.abs(positions.getX(33) + .1) < 1e-6, 'mirrored image maps to a negative local X');
assert.ok(Math.abs(positions.getY(33) - .025) < 1e-6, 'Y uses source height');
assert.ok(Math.abs(positions.getZ(33) - .008) < 1e-6, 'depth shell is inset by 2 mm at the eye');
for (const [pitch, yaw, roll] of [[0, 0, 0], [.25, .5, -.3], [-.4, -.65, .2]]) {
  const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));
  updateFaceOccluder(face.geometry, landmarks, 1000, 500, 1, 1000, rotation, 1);
  const local = new THREE.Vector3().fromBufferAttribute(face.geometry.getAttribute('position'), 33);
  const pose = new THREE.Group();
  pose.quaternion.copy(rotation);
  pose.scale.set(-1000, 1000, 1000);
  pose.updateMatrixWorld(true);
  const projected = local.applyMatrix4(pose.matrixWorld);
  const expected = new THREE.Vector3(100, 25, 10).add(new THREE.Vector3(0, 0, -2).applyQuaternion(rotation).multiplyScalar(1000 / 1000));
  assert.ok(projected.distanceTo(expected) < .001, `face reprojection survives pitch ${pitch}, yaw ${yaw}, roll ${roll}`);
}
// After tracking loss, the next detected face must replace all old vertices.
landmarks[33] = { x: .6, y: .6, z: .02 };
updateFaceOccluder(face.geometry, landmarks, 1000, 500, 1, 1000, new THREE.Quaternion(), 1);
assert.ok(Math.abs(positions.getX(33) - .1) < 1e-6, 'reacquisition can discard stale face geometry');
face.geometry.dispose();
face.depth.material.dispose();
face.debug.material.dispose();
face.shadow.material.dispose();
const skullBuffer = readFileSync(new URL('../../public/models/head-occluder.glb', import.meta.url));
const skullSource = await new GLTFLoader().parseAsync(skullBuffer.buffer.slice(skullBuffer.byteOffset, skullBuffer.byteOffset + skullBuffer.byteLength), '');
const skull = createSkullOccluder(skullSource.scene);
assert.ok(skull.depth.children.length > 0, 'supplied skull GLB registers');
assert.ok(skull.root.scale.toArray().every(value => Number.isFinite(value) && value > 0));
assert.equal(skull.depth.children[0].material.colorWrite, false);
assert.equal(skull.debug.visible, false);
assert.equal(skull.shadow.children[0].material.depthWrite, false);
const visibility = {useFaceOcclusion:true,useSkullOcclusion:false,softShadow:true,showFaceOccluder:false,showSkull:true};
applyOcclusionVisibility(face,skull,visibility);
assert.equal(face.depth.visible,true);
assert.equal(face.shadow.visible,true);
assert.equal(skull.depth.visible,false,'uncalibrated skull cannot silently erase temple arms');
assert.equal(skull.shadow.visible,false,'disabled skull must not leave a shadow patch');
assert.equal(skull.debug.visible,true,'alignment overlay works before enabling skull depth');
applyOcclusionVisibility(face,skull,{...visibility,useFaceOcclusion:false,useSkullOcclusion:true});
assert.equal(face.depth.visible,false);
assert.equal(face.shadow.visible,false);
assert.equal(skull.depth.visible,true);
applyOcclusionVisibility(face,skull,{...visibility,useFaceOcclusion:false,softShadow:false});
assert.equal(face.depth.visible||skull.depth.visible,false,'diagnostic mode reveals all exported geometry');
for (const group of [skull.depth, skull.shadow]) {
  const material = group.children[0].material;
  const shader = {
    uniforms: {},
    vertexShader: '#include <begin_vertex>',
    fragmentShader: '#include <clipping_planes_fragment>',
  };
  material.onBeforeCompile(shader);
  assert.match(shader.fragmentShader, /if \(side < 0\.68 && front > 0\.46\) discard;/, 'skull depth and shadow omit the generic front surface');
  assert.ok(shader.uniforms.skullCenter && shader.uniforms.skullSize);
}
console.log(`Face occlusion topology: ${topology.vertices.length} vertices, ${topology.indices.length / 3} triangles; supplied skull registration and depth checks passed.`);
