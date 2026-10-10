// Local QA entry point, separate from the merchant app. No project mutations.
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {preservePhotoTextureColors} from './viewer/materialPolicy';
import {testPhotoMaterialPolicy} from './viewer/materialPolicy.test';
const select=(id:string)=>document.getElementById(id) as HTMLSelectElement;
const result=document.getElementById('result')!;
const scenes=['before','after'].map(id=>{
 const canvas=document.getElementById(id) as HTMLCanvasElement;
 const renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
 renderer.setSize(780,480,false);renderer.outputColorSpace=THREE.SRGBColorSpace;
 renderer.toneMapping=THREE.NoToneMapping;
 const scene=new THREE.Scene();scene.background=new THREE.Color('#f4f6f5');
 scene.add(new THREE.HemisphereLight(0xffffff,0x63756d,2));
 const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(1,2,4);scene.add(light);
 const camera=new THREE.PerspectiveCamera(32,780/480,.0001,10);
 return {canvas,renderer,scene,camera,model:null as THREE.Group|null};
});
let epoch=0;
function dispose(root:THREE.Object3D){
 const textures=new Set<THREE.Texture>();
 root.traverse(n=>{if(n instanceof THREE.Mesh){n.geometry.dispose();(Array.isArray(n.material)?n.material:[n.material]).forEach(m=>{Object.values(m).forEach(v=>{if(v instanceof THREE.Texture)textures.add(v);});m.dispose();});}});
 textures.forEach(t=>t.dispose());
}
async function load(){
 const generation=++epoch;
 result.textContent='Loading pinned GLBs…';
 try{
  const assets=await Promise.allSettled(['/fidelity-qa/baseline.glb',`/fidelity-qa/${select('variant').value}.glb`].map(url=>new GLTFLoader().loadAsync(url)));
  if(generation!==epoch||assets.some(a=>a.status==='rejected')){
   assets.forEach(a=>{if(a.status==='fulfilled')dispose(a.value.scene);});
   if(generation!==epoch)return;
   throw new Error('One comparison GLB could not be loaded.');
  }
  const roots=assets.map(a=>{if(a.status!=='fulfilled')throw new Error('Missing GLB');return a.value;});
  roots.forEach(({scene:root},index)=>{
   const entry=scenes[index];
   if(entry.model){entry.scene.remove(entry.model);dispose(entry.model);}
   if(select('surface').value==='photo')preservePhotoTextureColors(root);
   else {
    const textures=new Set<THREE.Texture>();
    root.traverse(n=>{if(n instanceof THREE.Mesh){
     (Array.isArray(n.material)?n.material:[n.material]).forEach(m=>{Object.values(m).forEach(v=>{if(v instanceof THREE.Texture)textures.add(v);});m.dispose();});
     n.visible=!n.name.startsWith('lens_');
     n.material=select('surface').value==='clay'?new THREE.MeshStandardMaterial({color:'#929d97',roughness:.8,side:THREE.DoubleSide,flatShading:!n.geometry.getAttribute('normal')}):new THREE.MeshBasicMaterial({color:'#111111',side:THREE.DoubleSide});
    }});
    textures.forEach(t=>t.dispose());
   }
   entry.model=root;entry.scene.add(root);
  });
  render();result.textContent=`Loaded both GLBs. View: ${select('angle').value}. Surface: ${select('surface').value}. Candidate: ${select('variant').value}. No accuracy score inferred from this display.`;
 }catch(e){result.textContent=`LOAD FAILED: ${String(e)}`;}
}
function render(){
 const view=select('angle').value;
 const target=view==='hinge'?new THREE.Vector3(-.063,.013,-.010):new THREE.Vector3(0,-.004,-.058);
 const offsets:Record<string,THREE.Vector3>={front:new THREE.Vector3(0,0,.36),side:new THREE.Vector3(-.36,0,0),threequarter:new THREE.Vector3(-.19,.055,.30),top:new THREE.Vector3(0,.36,.0001),hinge:new THREE.Vector3(-.066,.011,.075)};
 for(const e of scenes){e.camera.position.copy(target).add(offsets[view]);e.camera.up.set(0,1,0);e.camera.lookAt(target);e.renderer.render(e.scene,e.camera);}
}
select('angle').onchange=()=>{render();result.textContent=`Same camera: ${select('angle').value}`;};
select('surface').onchange=load;select('variant').onchange=load;
document.getElementById('verify')!.onclick=async()=>{
 result.textContent='Checking GLB reload, transforms, independent texture references and lens materials…';
 try{await testPhotoMaterialPolicy(`/fidelity-qa/${select('variant').value}.glb`);result.textContent='PASS: GLB reload and shared photo-color conversion preserve component transforms, texture references, frame colors and independent lens tint/opacity. This does not verify live-camera fit.';}
 catch(e){result.textContent=`MATERIAL CHECK FAILED: ${String(e)}`;}
};
document.getElementById('save')!.onclick=()=>{
 render();const canvas=document.createElement('canvas');canvas.width=1560;canvas.height=520;const ctx=canvas.getContext('2d')!;
 ctx.fillStyle='#ffffff';ctx.fillRect(0,0,1560,520);ctx.fillStyle='#19312a';ctx.font='22px sans-serif';ctx.fillText('Before',12,28);ctx.fillText(`After: ${select('variant').value} (${select('surface').value})`,792,28);
 scenes.forEach((e,i)=>ctx.drawImage(e.canvas,i*780,40));const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=`fidelity-${select('variant').value}-${select('angle').value}-${select('surface').value}.png`;a.click();
};
void load();
