import {useEffect,useRef,useState} from 'react';
import {api} from '../api';
import type {Annotation,Point,Project,AnatomyReport} from '../types';
import AnatomySummary from './AnatomySummary';

const PARTS=['front','front-support','hinge-left','hinge-right','lens-mount-left','lens-mount-right','lens-left','lens-right','bridge-opening','bridge-hardware','nose-pad-left','nose-pad-right','temple-left','temple-right','shield','top'] as const;
const COLORS:Record<string,string>={'front-support':'#b58116','hinge-left':'#916ae0','hinge-right':'#916ae0','lens-mount-left':'#21a7a5','lens-mount-right':'#21a7a5','front':'#55ddbb','lens-left':'#69aafc','lens-right':'#f5ac72','bridge-opening':'#e273da','bridge-hardware':'#dba825','nose-pad-left':'#168bdb','nose-pad-right':'#168bdb','temple-left':'#b89bff','temple-right':'#f180b8','shield':'#f0cf63','top':'#72d9ec'};
const ANCHORS:Record<string,string[]>={'hinge-left':['hinge'],'hinge-right':['hinge'],top:['bridge','hinge-left','hinge-right','tip-left','tip-right'],'temple-left':['hinge','tip'],'temple-right':['hinge','tip'],front:['bridge','hinge-left','hinge-right','nose-pad-left','nose-pad-right'],shield:['bridge','hinge-left','hinge-right'],'lens-left':['nose-pad-left'],'lens-right':['nose-pad-right']};
function defaultPart(view:string,category:Project['category']){return view==='top'?'top':view.includes('left')&&!view.includes('front')?'temple-left':view.includes('right')&&!view.includes('front')?'temple-right':category==='sports-shield'?'shield':category==='rimless'?'lens-left':'front';}
const GUIDANCE:Record<Project['category'],string>={
 'opaque-full-rim':'Trace the outer front and both lens openings. Add side temple profiles to replace inferred arms.',
 'thin-metal':'Trace the outer metal front and both lens openings. Keep narrow wire contours aligned with the photo.',
 'transparent-full-rim':'Trace the outer clear frame and both lens openings; low contrast edges often need manual correction.',
 'semi-rimless':'Trace the front’s upper rim and both lens boundaries. The lower support is modeled separately.',
 'rimless':'Trace both lens edges precisely. Review bridge hardware and nose pads as separate parts. A front outer frame outline is not required.',
 'sports-shield':'Use Shield for one continuous lens. For dual-lens wraparound glasses, trace Front and both lens openings.'
};
type Props={project:Project;onUpdate:(project:Project)=>void;onError:(message:string)=>void;initialPhotoId?:string};
type Mode='select'|'draw'|'anchor'|'add'|'erase';
const copy=(a:Annotation[])=>structuredClone(a);

export default function OutlineEditor({project,onUpdate,onError,initialPhotoId}:Props){
 const [photoId,setPhotoId]=useState(initialPhotoId||project.photos.find(p=>p.view==='front')?.id||project.photos[0]?.id||'');
 const [part,setPart]=useState<string>(()=>defaultPart(project.photos.find(p=>p.id===(initialPhotoId||project.photos.find(p=>p.view==='front')?.id||project.photos[0]?.id))?.view||'front',project.category));
 const [mode,setMode]=useState<Mode>('select');
 const [anchorName,setAnchorName]=useState('bridge');
 const [draft,setDraft]=useState<Annotation[]>(copy(project.annotations));
 const [warnings,setWarnings]=useState<string[]>([]);
 const [anatomy,setAnatomy]=useState<AnatomyReport|null>(null);
 const [zoom,setZoom]=useState(1);
 const [analysisBackend,setAnalysisBackend]=useState<'cpu-metal'|'heuristics'|'sam3'|'sam2'>('cpu-metal');
 const [busy,setBusy]=useState(false);
 const [dirty,setDirty]=useState(false);
 const [drag,setDrag]=useState<number|null>(null);
 const [dragAnchor,setDragAnchor]=useState<string|null>(null);
 const [brushRadius,setBrushRadius]=useState(12);
 const [paintPreview,setPaintPreview]=useState<Point[]>([]);
 const stroke=useRef<Point[]|null>(null);
 const history=useRef<Annotation[][]>([]);
 const future=useRef<Annotation[][]>([]);
 const svg=useRef<SVGSVGElement>(null);
 const photo=project.photos.find(p=>p.id===photoId);
 const current=draft.find(a=>a.photo_id===photoId&&a.part===part);
 useEffect(()=>{setDraft(copy(project.annotations));setDirty(false);history.current=[];future.current=[];},[project.id,project.annotations]);
 useEffect(()=>{if(initialPhotoId&&project.photos.some(p=>p.id===initialPhotoId))setPhotoId(initialPhotoId);},[initialPhotoId,project.photos]);
 useEffect(()=>{if(!project.photos.some(p=>p.id===photoId))setPhotoId(project.photos.find(p=>p.view==='front')?.id||project.photos[0]?.id||'');},[project.photos,photoId]);
 useEffect(()=>{setAnatomy(null);if(photo)setPart(defaultPart(photo.view,project.category));},[photoId,project.category]);
 function remember(){history.current.push(copy(draft));if(history.current.length>50)history.current.shift();future.current=[];setDirty(true);}
 function updateAnnotation(updater:(a:Annotation)=>Annotation){
  setAnatomy(null);
  remember();setDraft(existing=>{
   const index=existing.findIndex(a=>a.photo_id===photoId&&a.part===part);
   const base:Annotation=index>=0?existing[index]:{photo_id:photoId,part,points:[],anchors:{},approved:false,source:'manual'};
   const next=existing.slice();const updated={...updater(base),approved:false,source:'manual',evidence:null};
   if(index>=0)next[index]=updated;else next.push(updated);
   return next;
  });
 }
 function pointFromEvent(e:React.PointerEvent<SVGElement>):Point|null{
  const matrix=svg.current?.getScreenCTM();if(!matrix||!photo)return null;
  const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix.inverse());
  return {x:Math.max(0,Math.min(photo.width,Math.round(p.x*10)/10)),y:Math.max(0,Math.min(photo.height,Math.round(p.y*10)/10))};
 }
 function canvasDown(e:React.PointerEvent<SVGSVGElement>){
  if(!photo||busy)return;const p=pointFromEvent(e);if(!p)return;
  if(mode==='add'||mode==='erase'){stroke.current=[p];setPaintPreview([p]);e.currentTarget.setPointerCapture(e.pointerId);return;}
  if(mode==='draw')updateAnnotation(a=>({...a,points:[...a.points,p]}));
  if(mode==='anchor'&&anchorName.trim())updateAnnotation(a=>({...a,anchors:{...a.anchors,[anchorName.trim()]:p}}));
 }
 function move(e:React.PointerEvent<SVGElement>){
  if(stroke.current){const p=pointFromEvent(e);if(p&&stroke.current.length<2000){stroke.current.push(p);setPaintPreview([...stroke.current]);}return;}
  if(drag===null&&dragAnchor===null)return;const p=pointFromEvent(e);if(!p)return;
  setDraft(existing=>existing.map(a=>a.photo_id===photoId&&a.part===part?{...a,points:drag===null?a.points:a.points.map((v,i)=>i===drag?p:v),anchors:dragAnchor===null?a.anchors:{...a.anchors,[dragAnchor]:p},approved:false,source:'manual'}:a));
 }
 function startVertex(e:React.PointerEvent<SVGCircleElement>,index:number){if(mode==='add'||mode==='erase')return;e.stopPropagation();remember();setDrag(index);e.currentTarget.setPointerCapture(e.pointerId);}
 function startAnchor(e:React.PointerEvent<SVGCircleElement>,name:string){if(mode==='add'||mode==='erase')return;e.stopPropagation();remember();setDragAnchor(name);e.currentTarget.setPointerCapture(e.pointerId);}
 async function stopDrag(){
  setDrag(null);setDragAnchor(null);
  const painted=stroke.current;stroke.current=null;setPaintPreview([]);
  if(!painted||!photo||(mode!=='add'&&mode!=='erase'))return;
  setBusy(true);
  try{
   const response=await fetch('/api/masks/refine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({width:photo.width,height:photo.height,points:current?.points||[],strokes:[{mode,radius:brushRadius,points:painted}]})});
   const result=await response.json();if(!response.ok)throw new Error(result.detail||'Brush correction failed');
   updateAnnotation(a=>({...a,points:result.points}));setWarnings(result.warnings||[]);
  }catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 function undo(){const previous=history.current.pop();if(!previous)return;future.current.push(copy(draft));setDraft(previous);setAnatomy(null);setDirty(true);}
 function redo(){const next=future.current.pop();if(!next)return;history.current.push(copy(draft));setDraft(next);setAnatomy(null);setDirty(true);}
 async function extract(){if(!photo)return;if(photo.role==='validation'){onError('Validation photos are held out from fitting. Select a fitting photo for extraction.');return;}setBusy(true);try{
  const result=await api.extract(project.id,photo.id,analysisBackend);
  setWarnings(result.warnings);
  setAnatomy(result.anatomy??null);
  remember();setDraft(existing=>[...existing.filter(a=>a.photo_id!==photo.id),...result.annotations.map(a=>({...a,approved:false}))]);
 }catch(e){const message=String(e instanceof Error?e.message:e);setWarnings([message]);onError(message);}finally{setBusy(false);}}
 async function save(approve:boolean){
  if(!photo)return;
  if(approve&&photo.role==='validation'){onError('Validation photos are withheld from reconstruction. Select a fitting photo to approve geometry.');return;}
  const annotations=draft.map(a=>a.photo_id===photoId&&approve?{...a,approved:a.points.length>=3}:a);
  if(approve&&!annotations.some(a=>a.photo_id===photoId&&a.points.length>=3)){onError('Draw at least one closed outline with three points before approval.');return;}
  const top=annotations.find(a=>a.photo_id===photoId&&a.part==='top'&&a.approved);
  const missingTop=top?['bridge','hinge-left','hinge-right'].filter(name=>!top.anchors[name]):[];
  if(approve&&missingTop.length){onError(`Place ${missingTop.join(', ')} anchor${missingTop.length===1?'':'s'} on the top photo before approval.`);return;}
  setBusy(true);try{
   if(approve){const audit=await api.auditAnatomy(project.id,photoId,annotations.filter(a=>a.photo_id===photoId),project.category);setAnatomy(audit);if(audit.blocking)throw new Error('Structure checks failed. Correct the listed geometry findings before approval.');}
   onUpdate(await api.patch(project.id,{annotations}));setDirty(false);
  }
  catch(e){onError(String(e instanceof Error?e.message:e));}finally{setBusy(false);}
 }
 const selectedApproved=draft.some(a=>a.photo_id===photoId&&a.approved);
 return <section className="step-content editor-page">
  <div className="section-heading"><div><span className="eyebrow">02 / Geometry evidence</span><h2>Trace & approve outlines</h2><p>Extract a starting trace, then correct it against the image. All points use the original photo’s pixel coordinates.</p><p className="category-guidance">{GUIDANCE[project.category]}</p></div></div>
  {!photo?<div className="empty-panel"><span>◌</span><h3>No photos to trace yet</h3><p>Add a front photo in Capture, then return here to draw the frame and lens boundaries.</p></div>:<>
   <div className="editor-toolbar"><label>Photo<select value={photoId} onChange={e=>{setPhotoId(e.target.value);setZoom(1);}}>{project.photos.map(p=><option value={p.id} key={p.id}>{p.view} · {p.width} × {p.height}{p.role==='validation'?' · validation':''}</option>)}</select></label><label>Analysis<select aria-label="Outline analysis backend" value={analysisBackend} onChange={e=>setAnalysisBackend(e.target.value as typeof analysisBackend)} disabled={busy}><option value="cpu-metal">CPU · thin metal + contours</option><option value="heuristics">CPU · standard contours</option><option value="sam3">SAM 3 · CPU</option><option value="sam2">SAM 2.1</option></select></label><button className="button secondary" onClick={extract} disabled={busy||photo.role==='validation'}>{busy?'Working…':'Extract contours'}</button><span className="toolbar-spacer"/><button className="button quiet" onClick={undo} disabled={!history.current.length}>↶ Undo</button><button className="button quiet" onClick={redo} disabled={!future.current.length}>↷ Redo</button><button className="button quiet" onClick={()=>setZoom(z=>Math.max(1,z/1.25))} disabled={zoom<=1}>−</button><span className="zoom-label">{Math.round(zoom*100)}%</span><button className="button quiet" onClick={()=>setZoom(z=>Math.min(8,z*1.25))} disabled={zoom>=8}>＋</button></div>
    <div className="editor-workspace"><aside className="editor-rail"><h3>Parts</h3><p>Select what you’re tracing</p>{PARTS.map(name=><button className={`part-button ${part===name?'active':''}`} onClick={()=>setPart(name)} key={name}><span style={{background:COLORS[name]}}/>{name.replaceAll('-',' ')}<small>{draft.find(a=>a.photo_id===photoId&&a.part===name)?.points.length||0}</small></button>)}<div className="rail-note">{GUIDANCE[project.category]}</div></aside>
    <div className="editor-center">
     <div className="canvas-head"><strong>{photo.view}</strong><span>{photo.width} × {photo.height} px · {photo.role}</span></div>
     <div className="canvas-scroll"><div className="canvas-stage" style={{width:`${zoom*100}%`}}>
      <svg ref={svg} viewBox={`0 0 ${photo.width} ${photo.height}`} onPointerDown={canvasDown} onPointerMove={move} onPointerUp={stopDrag} onPointerCancel={stopDrag} role="img" aria-label={`Outline editing canvas for ${photo.view}`}>
       <image href={photo.url} x="0" y="0" width={photo.width} height={photo.height}/>
       {draft.filter(a=>a.photo_id===photoId).map(a=><g key={a.part} opacity={a.part===part?1:.58}>
        <polygon points={a.points.map(p=>`${p.x},${p.y}`).join(' ')} fill={a.points.length>=3?`${COLORS[a.part]||'#55ddbb'}27`:'none'} stroke={COLORS[a.part]||'#55ddbb'} strokeWidth={Math.max(2,photo.width/500)} strokeLinejoin="round"/>
        {Object.entries(a.anchors).map(([name,p])=><g key={name}><circle cx={p.x} cy={p.y} r={Math.max(5,photo.width/160)} fill="#fff" stroke={COLORS[a.part]||'#55ddbb'} strokeWidth="2" className="anchor-handle" onPointerDown={e=>{e.stopPropagation();if(a.part===part)startAnchor(e,name);}}/><text x={p.x+8} y={p.y-8} fill="#fff" stroke="#172436" strokeWidth="2" paintOrder="stroke" fontSize={Math.max(11,photo.width/95)} pointerEvents="none">{name}</text></g>)}
        {a.part===part&&a.points.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r={Math.max(5,photo.width/160)} fill="#fff" stroke={COLORS[a.part]} strokeWidth="3" className="vertex" onPointerDown={e=>startVertex(e,i)}/>)}
       </g>)}
       {paintPreview.length>0&&<polyline points={paintPreview.map(p=>`${p.x},${p.y}`).join(' ')} fill="none" stroke={mode==='erase'?'#f36c75aa':'#55ddbbaa'} strokeWidth={brushRadius*2} strokeLinecap="round" strokeLinejoin="round" pointerEvents="none"/>}
      </svg>
     </div></div>
     <div className="canvas-foot">Click to add points in Draw mode. Choose an anchor, then click the image to place it. Drag handles to correct either.</div>
    </div>
    <aside className="editor-controls"><h3>Editing tools</h3><div className="mode-switch"><button className={mode==='select'?'selected':''} onClick={()=>setMode('select')}>Move points</button><button className={mode==='draw'?'selected':''} onClick={()=>setMode('draw')}>Draw</button><button className={mode==='anchor'?'selected':''} onClick={()=>setMode('anchor')}>Anchor</button></div>{mode==='anchor'&&<div className="anchor-tools"><strong>Choose anchor, then click image</strong><div className="anchor-presets">{(ANCHORS[part]||['bridge','hinge','tip']).map(name=><button key={name} className={anchorName===name?'selected':''} onClick={()=>setAnchorName(name)} aria-pressed={anchorName===name}>{name}{current?.anchors[name]?' ✓':''}</button>)}</div><label className="field">Custom anchor name<input value={anchorName} onChange={e=>setAnchorName(e.target.value)} placeholder="Anchor name"/></label>{current?.anchors[anchorName]&&<button className="button quiet full" onClick={()=>updateAnnotation(a=>{const anchors={...a.anchors};delete anchors[anchorName];return {...a,anchors};})}>Remove selected anchor</button>}</div>}
     <div className="outline-details"><h4>{part.replaceAll('-',' ')}</h4><p>{current?.points.length||0} vertices · {Object.keys(current?.anchors||{}).length} anchors</p><p>{current?.approved?'Approved for generation':'Unapproved trace'}</p></div>
     <div className="mode-switch"><button className={mode==='add'?'selected':''} onClick={()=>setMode('add')} disabled={busy}>Brush add</button><button className={mode==='erase'?'selected':''} onClick={()=>setMode('erase')} disabled={busy}>Brush erase</button></div>
     {(mode==='add'||mode==='erase')&&<label className="field">Brush radius · {brushRadius} px<input type="range" min="2" max="80" value={brushRadius} onChange={e=>setBrushRadius(Number(e.target.value))}/><small>Paint the boundary of one connected part. Lens holes are separate outlines.</small></label>}
     <button className="button quiet full" onClick={()=>updateAnnotation(a=>({...a,points:[]}))} disabled={!current?.points.length}>Clear points</button><button className="button quiet full" onClick={()=>updateAnnotation(a=>({...a,points:a.points.slice(0,-1)}))} disabled={!current?.points.length}>Remove last point</button>
     <div className="approval-box"><strong>Approve evidence</strong><p>{photo.role==='validation'?'Validation photos stay out of reconstruction. You can save a trace for visual comparison.':'Review contours before generating. Editing an approved outline returns it to unapproved status.'}</p><button className="button primary full" onClick={()=>save(true)} disabled={busy||photo.role==='validation'}>{busy?'Saving…':'Save & approve photo'}</button><button className="button secondary full" onClick={()=>save(false)} disabled={busy||!dirty}>Save draft</button></div>
    </aside></div>
   {warnings.length>0&&<div className="notice warning"><strong>Extraction notes</strong><ul>{warnings.map((w,i)=><li key={i}>{w}</li>)}</ul></div>}
   <AnatomySummary report={anatomy}/>
   {selectedApproved&&!dirty&&<div className="notice success">At least one outline on this photo is approved and saved.</div>}
  </>}
 </section>;
}
