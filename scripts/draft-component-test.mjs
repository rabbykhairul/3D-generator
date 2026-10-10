import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import ts from 'typescript';

const asModule=code=>'data:text/javascript;base64,'+Buffer.from(code).toString('base64');
const hooks={slots:[],cursor:0};
const calls={extract:[],patch:[]};
const point=(x,y)=>({x,y});
const annotation=(photo_id,part,points,source,approved=false)=>({photo_id,part,points,anchors:{},source,approved});
const oldFront=annotation('front','front',[point(0,0),point(100,0),point(100,50)],'manual',true);
const freshFront=annotation('front','front',[point(10,10),point(90,10),point(90,40)],'automatic',true);
const leftLens=annotation('front','lens-left',[point(15,15),point(40,15),point(40,35)],'automatic');
const rightLens=annotation('front','lens-right',[point(60,15),point(85,15),point(85,35)],'automatic');
const sideArm=annotation('left','temple-left',[point(5,5),point(80,5),point(80,20)],'side-extraction',true);
const mirrored=annotation('left','temple-right',[point(5,5),point(80,5),point(80,20)],'explicit-mirrored-side-assumption:manual',true);
const unrelated=annotation('front','bridge-opening',[point(1,1),point(3,1),point(3,3)],'manual',true);
const frontPhoto={id:'front',view:'front',role:'fitting',url:'',width:100,height:50,warnings:[]};
const sidePhoto={id:'left',view:'left',role:'fitting',url:'',width:100,height:50,warnings:[]};
const measurements=Object.fromEntries(['frame_width','lens_width','lens_height','bridge_width','temple_length'].map(k=>[k,{value:100,source:'manual'}]));
const project={id:'project',revision:1,category:'opaque-full-rim',photos:[frontPhoto,sidePhoto],annotations:[oldFront,mirrored,unrelated],measurements};
let frontConstruction={category:'thin-metal',confidence:'high',reason:'Visible narrow rims.'};
let extractResponse=null;
const api={
 async extract(id,photoId,backend){calls.extract.push(backend?[id,photoId,backend]:[id,photoId]);if(extractResponse)return extractResponse();return photoId==='front'?{annotations:[freshFront,leftLens,rightLens],warnings:['Fresh front'],construction:frontConstruction}:{annotations:[sideArm],warnings:['Fresh side'],construction:{category:null,confidence:'low',reason:'Side photo.'}};},
 async auditAnatomy(){return {blocking:false,issues:[],nodes:[]};},
 async patch(id,data){calls.patch.push([id,data]);return {...project,...data,revision:project.revision+1};}
};
globalThis.__draftTest={hooks,api};
const react=asModule(`export function useRef(initial){const h=globalThis.__draftTest.hooks,i=h.cursor++;return h.slots[i]??(h.slots[i]={current:initial});}
export function useState(initial){const h=globalThis.__draftTest.hooks,i=h.cursor++;if(!(i in h.slots))h.slots[i]=initial;const slots=h.slots;return [slots[i],value=>{slots[i]=typeof value==='function'?value(slots[i]):value;}];}
export function useEffect(){}`);
const jsx=asModule('export const Fragment=Symbol.for("react.fragment");export const jsx=(type,props)=>({type,props:props||{}});export const jsxs=jsx;');
const apiModule=asModule('export const api=globalThis.__draftTest.api;');
const dimensions=asModule('export const approximateMeasurements=(draft,measurements)=>measurements;');
const diagnosticComponent=asModule('export default function Diagnostic(){return null;}');
async function loadComponent(path){
 const source=await fs.readFile(path,'utf8');
 let code=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2022}}).outputText;
 code=code.replaceAll('"react/jsx-runtime"',JSON.stringify(jsx)).replaceAll("'react'",JSON.stringify(react)).replaceAll("'../api'",JSON.stringify(apiModule)).replaceAll("'../draftMeasurements'",JSON.stringify(dimensions));
 code=code.replaceAll("'./DetectionStatus'",JSON.stringify(diagnosticComponent)).replaceAll("'./AnatomySummary'",JSON.stringify(diagnosticComponent));
 return (await import(asModule(code))).default;
}
function reset(){hooks.slots=[];hooks.cursor=0;calls.extract=[];calls.patch=[];}
function render(Component,props){hooks.cursor=0;return Component(props);}
function children(node){return Array.isArray(node)?node:node&&typeof node==='object'?[node.props?.children]:[];}
function walk(node){if(Array.isArray(node))return node.flatMap(walk);if(!node||typeof node!=='object')return [];return [node,...walk(children(node))];}
function label(node){return walk(node.props?.children).map(x=>label(x)).join('')+(typeof node.props?.children==='string'?node.props.children:'');}
function button(tree,name){const found=walk(tree).find(n=>n.type==='button'&&label(n)===name);assert.ok(found,`Missing button: ${name}`);return found;}
function polygon(tree){return walk(tree).find(n=>n.type==='polygon');}
function constructionSelect(tree){const found=walk(tree).find(n=>n.type==='select'&&n.props['aria-label']==='Construction to use for this draft');assert.ok(found,'Missing construction selection');return found;}
function mode(tree,name){const found=walk(tree).find(n=>n.type==='label'&&n.props.className==='construction-mode'&&JSON.stringify(n.props.children).includes(name));assert.ok(found,`Missing mode: ${name}`);return walk(found).find(n=>n.type==='input');}

const QuickDraft=await loadComponent('frontend/src/components/QuickDraft.tsx');
const SideDraft=await loadComponent('frontend/src/components/SideDraft.tsx');
let updates=[],edits=[],generations=[];
const common={project,onUpdate:p=>updates.push(p),onError:()=>{},onEditPhoto:id=>edits.push(id),onGenerate:p=>generations.push(p),autoAnalyze:false};
const frontProps={...common,onDimensions:()=>{},onAnalysisStarted:()=>{}};

reset();
let tree=render(QuickDraft,frontProps);
button(tree,'Show approved outlines').props.onClick();
tree=render(QuickDraft,frontProps);
assert.equal(polygon(tree).props.points,'0,0 100,0 100,50');
assert.equal(calls.extract.length,0);
await button(tree,'Analyze current front photo').props.onClick();
tree=render(QuickDraft,frontProps);
assert.deepEqual(calls.extract,[['project','front','sam3']]);
assert.equal(polygon(tree).props.points,'10,10 90,10 90,40');
assert.equal(calls.patch.length,0,'Extraction must not save or replace reviewed outlines');
assert.equal(project.annotations[0].source,'manual');
assert.equal(constructionSelect(tree).props.value,'thin-metal');
assert.equal(constructionSelect(tree).props.disabled,true);
await button(tree,'Approve outlines & generate draft').props.onClick();
assert.equal(calls.patch.length,1);
assert.equal(calls.patch[0][1].category,'thin-metal','Suggested construction is saved only on approval');
assert.equal(calls.patch[0][1].annotations.find(a=>a.part==='front').approved,true);
assert.equal(generations.length,1);

for(const confidence of ['high','medium']){
 reset();updates=[];generations=[];
 frontConstruction={category:'rimless',confidence,reason:'No visible continuous rim.'};
 tree=render(QuickDraft,frontProps);
 await button(tree,'Analyze current front photo').props.onClick();
 tree=render(QuickDraft,frontProps);
 assert.equal(calls.patch.length,0,`${confidence} analysis must not mutate the project`);
 assert.equal(constructionSelect(tree).props.value,'rimless');
 assert.equal(button(tree,'Approve outlines & generate draft').props.disabled,false);
 await button(tree,'Approve outlines & generate draft').props.onClick();
 assert.equal(calls.patch[0][1].category,'rimless');
}

for(const category of ['thin-metal',null]){
 reset();updates=[];generations=[];
 frontConstruction={category,confidence:'low',reason:'Rim obscured.'};
 tree=render(QuickDraft,frontProps);
 await button(tree,'Analyze current front photo').props.onClick();
 tree=render(QuickDraft,frontProps);
 assert.equal(constructionSelect(tree).props.value,'');
 assert.equal(button(tree,'Approve outlines & generate draft').props.disabled,true,'Uncertain construction requires an explicit choice');
 assert.equal(calls.patch.length,0);
 mode(tree,'Choose manually').props.onChange();
 tree=render(QuickDraft,frontProps);
 assert.equal(constructionSelect(tree).props.value,'opaque-full-rim','Manual selection starts from saved category');
 constructionSelect(tree).props.onChange({target:{value:'transparent-full-rim'}});
 tree=render(QuickDraft,frontProps);
 await button(tree,'Approve outlines & generate draft').props.onClick();
 assert.equal(calls.patch[0][1].category,'transparent-full-rim','Manual override wins');
}

reset();updates=[];generations=[];
frontConstruction={category:'thin-metal',confidence:'high',reason:'Visible narrow rims.'};
tree=render(QuickDraft,frontProps);
await button(tree,'Analyze current front photo').props.onClick();
tree=render(QuickDraft,frontProps);
mode(tree,'Choose manually').props.onChange();
tree=render(QuickDraft,frontProps);
constructionSelect(tree).props.onChange({target:{value:'transparent-full-rim'}});
tree=render(QuickDraft,frontProps);
await button(tree,'Analyze current front photo').props.onClick();
tree=render(QuickDraft,frontProps);
assert.equal(constructionSelect(tree).props.value,'transparent-full-rim','Reanalysis preserves a manual override');
button(tree,'Show approved outlines').props.onClick();
tree=render(QuickDraft,frontProps);
assert.equal(constructionSelect(tree).props.value,'opaque-full-rim','Showing reviewed outlines uses the saved category');
assert.equal(calls.patch.length,0);
reset();
const anotherProject={...project,id:'other-project',revision:2,photos:[{...frontPhoto,id:'other-front'}],annotations:[]};
tree=render(QuickDraft,{...frontProps,project:anotherProject});
assert.equal(walk(tree).some(n=>n.props?.className==='construction-review'),false,'A remounted project/photo has no stale suggestion');

reset();
let finishExtract;
extractResponse=()=>new Promise(resolve=>{finishExtract=resolve;});
tree=render(QuickDraft,frontProps);
const pending=button(tree,'Analyze current front photo').props.onClick();
tree=render(QuickDraft,{...frontProps,project:anotherProject});
finishExtract({annotations:[freshFront,leftLens,rightLens],warnings:[],construction:frontConstruction});
await pending;
tree=render(QuickDraft,{...frontProps,project:anotherProject});
assert.equal(polygon(tree),undefined,'Late analysis from a previous project/photo is discarded');
assert.equal(calls.patch.length,0);
extractResponse=null;

reset();updates=[];edits=[];generations=[];
tree=render(QuickDraft,frontProps);
await button(tree,'Analyze current front photo').props.onClick();
tree=render(QuickDraft,frontProps);
await button(tree,'Correct in Advanced').props.onClick();
assert.equal(calls.patch.length,1,'Correction explicitly saves draft suggestions');
assert.equal(calls.patch[0][1].annotations.find(a=>a.part==='front').approved,false);
assert.deepEqual(edits,['front']);

reset();updates=[];edits=[];generations=[];
const sideProps={...common,photo:sidePhoto,onStarted:()=>{}};
tree=render(SideDraft,sideProps);
assert.equal(button(tree,'Approve arm & regenerate').props.disabled,true);
await button(tree,'Adjust outline / hinge / tip').props.onClick();
assert.equal(calls.patch.length,0,'Opening the editor without a suggestion must not save');
assert.deepEqual(edits,['left']);
edits=[];
await button(tree,'Analyze side arm').props.onClick();
tree=render(SideDraft,sideProps);
assert.deepEqual(calls.extract,[['project','left','sam3']]);
assert.equal(calls.patch.length,0,'Side extraction must remain local until an explicit action');
await button(tree,'Adjust outline / hinge / tip').props.onClick();
assert.equal(calls.patch.length,1);
const adjusted=calls.patch[0][1].annotations;
assert.equal(adjusted.find(a=>a.photo_id==='left'&&a.part==='temple-left').approved,false);
assert.deepEqual(adjusted.find(a=>a.part==='temple-right'),mirrored,'Opposite mirrored arm is preserved');
assert.deepEqual(adjusted.find(a=>a.part==='front'),oldFront,'Other photo annotations are preserved');
assert.deepEqual(edits,['left']);

reset();updates=[];edits=[];generations=[];
tree=render(SideDraft,sideProps);
await button(tree,'Analyze side arm').props.onClick();
tree=render(SideDraft,sideProps);
await button(tree,'Approve arm & regenerate').props.onClick();
assert.equal(calls.patch.length,1,'Side approval is an explicit save');
assert.equal(calls.patch[0][1].annotations.find(a=>a.part==='temple-left').approved,true);
assert.ok(calls.patch[0][1].annotations.some(a=>a.part==='temple-right'&&a.source.startsWith('explicit-mirrored-side-assumption:')),'Re-analysis retains the existing explicit mirror choice');
assert.equal(generations.length,1);

reset();
const unmirroredProps={...sideProps,project:{...project,annotations:[oldFront,unrelated]}};
tree=render(SideDraft,unmirroredProps);
await button(tree,'Analyze side arm').props.onClick();
tree=render(SideDraft,unmirroredProps);
assert.equal(walk(tree).find(n=>n.type==='input'&&n.props.type==='checkbox').props.checked,false,'New photos never opt into mirroring automatically');

console.log('Draft component handler checks passed.');
