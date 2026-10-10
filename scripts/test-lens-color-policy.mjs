// Exercise the actual viewer policy without requiring a camera or WebGL.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import ts from 'typescript';
import {pathToFileURL} from 'node:url';
import * as THREE from 'three';
const source=fs.readFileSync('frontend/src/viewer/materialPolicy.ts','utf8');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext}}).outputText
  .replace("from 'three'",`from '${pathToFileURL(process.cwd()+'/node_modules/three/build/three.module.js')}'`);
const {preservePhotoTextureColors}=await import('data:text/javascript;base64,'+Buffer.from(compiled).toString('base64'));
const group=new THREE.Group();
const lens=new THREE.MeshStandardMaterial({color:new THREE.Color().setRGB(31/255,32/255,24/255),opacity:230/255,transparent:true});
lens.name='photo-estimated-lens';
const mesh=new THREE.Mesh(new THREE.PlaneGeometry(),lens);
const hardware=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial({color:'#888888'}));
const originalHardware=hardware.material;
group.add(mesh,hardware);
preservePhotoTextureColors(group);
assert(mesh.material instanceof THREE.MeshBasicMaterial);
assert.equal(mesh.material.toneMapped,false);
assert.equal(mesh.material.opacity,230/255);
assert.equal(mesh.material.side,THREE.FrontSide);
assert.equal(mesh.material.depthWrite,false);
assert(mesh.material.color.equals(lens.color));
assert.equal(hardware.material,originalHardware);
const displayed=mesh.material.color.clone().convertLinearToSRGB();
assert(Math.abs(displayed.r*255-98)<2 && Math.abs(displayed.g*255-100)<2);
console.log('PASS: photo-estimated lens uses linear GLB tint, unlit preview/AR policy, unchanged alpha; hardware remains lit.');
