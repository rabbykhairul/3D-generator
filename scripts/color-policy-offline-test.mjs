// Exercise the actual Three.js material policy without a server or GPU.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
import * as THREE from 'three';
const source=readFileSync(new URL('../frontend/src/viewer/materialPolicy.ts',import.meta.url),'utf8');
const js=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText
  .replace(/from ['"]three['"]/g,`from '${import.meta.resolve('three')}'`);
const {preservePhotoTextureColors}=await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
const root=new THREE.Group();
const map=new THREE.DataTexture(new Uint8Array([140,45,65,255]),1,1);map.colorSpace=THREE.SRGBColorSpace;
const entries=[['front_frame',{map,opacity:.8,transparent:true}],['hinge_right',{color:'#808080',metalness:.8}],
  ['lens_left',{color:'#123456',opacity:.7,transparent:true}],['lens_right',{name:'photo-estimated-lens',color:'#123456',opacity:.9,transparent:true}]];
const meshes=entries.map(([name,options])=>{
  const mesh=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshStandardMaterial(options));
  mesh.name=name;root.add(mesh);return mesh;
});
const old=meshes.map(m=>m.material);
preservePhotoTextureColors(root);
for(const i of [0,1,3]){
  assert(meshes[i].material instanceof THREE.MeshBasicMaterial);
  assert.equal(meshes[i].material.toneMapped,false);
  assert(meshes[i].material.color.equals(old[i].color));
  assert.equal(meshes[i].material.opacity,old[i].opacity);
}
assert.equal(meshes[0].material.map,map);
assert.equal(map.colorSpace,THREE.SRGBColorSpace);
assert.equal(meshes[2].material,old[2]);
assert.equal(meshes[3].material.side,THREE.FrontSide);
const gray=meshes[1].material.color.getHexString();assert.equal(gray,'808080');
const clearLens=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshBasicMaterial({name:'photo-estimated-lens',map,transparent:true}));
root.add(clearLens);preservePhotoTextureColors(root);
assert.equal(clearLens.material.alphaTest,.001); // keep estimated clear alpha (~.02)
assert.equal(clearLens.material.depthWrite,false);
console.log('Photo-color policy: frame textures, hardware, alpha and optical separation passed (offline; no pixel-render claim).');
