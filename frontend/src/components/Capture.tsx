import {useRef, useState} from 'react';
import {api} from '../api';
import type {Category,Photo, Project} from '../types';
import QuickDraft from './QuickDraft';
import SideDraft from './SideDraft';
import PhotoGuide from './PhotoGuide';
import {armPhotos} from '../photoEvidence';

const OVERVIEW = [
 ['front','Front','Straight on, temples open'],
 ['front-left','Front left','About 45° from front'],
 ['left','Left side','Show hinge and full temple'],
 ['rear-left','Rear left','About 45° from rear'],
 ['rear','Rear','Show both open temples'],
 ['rear-right','Rear right','About 45° from rear'],
 ['right','Right side','Show hinge and full temple'],
 ['front-right','Front right','About 45° from front'],
 ['top','Top','Reveal front curve and temple spread'],
 ['bottom','Bottom','Reveal lens and frame thickness'],
 ['elevated-front-left','Elevated left','Look down toward left hinge'],
 ['elevated-front-right','Elevated right','Look down toward right hinge'],
 ['inner','Inner / rear','Back surface evidence when registered; otherwise a comparison reference']
] as const;
const DETAIL = [['bridge','Bridge detail'],['hinge-left','Left hinge'],['hinge-right','Right hinge'],['nose-pads','Nose pads'],['inner-left','Inner left'],['inner-right','Inner right'],['detail','Other detail / scale']] as const;
const VIEW_USE:Record<string,string>={
 front:'Defines the approved front silhouette and lens openings.',
 left:'Can shape the left arm from an approved side outline.',
 right:'Can shape the right arm from an approved side outline.',
 'front-left':'Can check front-to-arm alignment when the views can be matched.',
 'front-right':'Can check front-to-arm alignment when the views can be matched.',
 inner:'Can inform the visible back surface when registered to approved geometry; otherwise used for comparison.'
};

type Props={project:Project;onUpdate:(project:Project)=>void;onError:(message:string)=>void;onEditPhoto:(photoId:string)=>void;onGenerate:(p:Project)=>void;onDimensions:()=>void;autoAnalyze:boolean;onAnalysisStarted:()=>void;sideRequest:string|null;onSideStarted:()=>void};

export default function Capture({project,onUpdate,onError,onEditPhoto,onGenerate,onDimensions,autoAnalyze,onAnalysisStarted,sideRequest,onSideStarted}:Props){
 const input=useRef<HTMLInputElement>(null);
 const [target,setTarget]=useState('front');
 const [role,setRole]=useState<'fitting'|'validation'>('fitting');
 const [busy,setBusy]=useState(false);
 const [detailView,setDetailView]=useState('bridge');
 const [advanced,setAdvanced]=useState(false);
 const photosFor=(view:string)=>project.photos.filter(p=>p.view===view).slice().reverse();
 function choose(view:string){setTarget(view);input.current?.click();}
 async function upload(file?:File){
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){onError('Use a JPEG, PNG or WebP image.');return;}
  if(file.size>20*1024*1024){onError('This image exceeds the 20 MB file limit.');return;}
  setBusy(true);
  try{onUpdate(await api.upload(project.id,file,target,role));}
  catch(e){onError(String(e instanceof Error?e.message:e));}
  finally{setBusy(false);if(input.current)input.current.value='';}
 }
 async function remove(photo:Photo){
  try{onUpdate(await api.removePhoto(project.id,photo.id));}
  catch(e){onError(String(e instanceof Error?e.message:e));}
 }
 async function changeCategory(category:Category){setBusy(true);try{onUpdate(await api.patch(project.id,{category}));}catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 async function relabel(photo:Photo,view:string){setBusy(true);try{onUpdate(await api.relabelPhoto(project.id,photo.id,view));}catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 const visible=advanced?OVERVIEW:['front','left','right','inner','front-left'].map(view=>OVERVIEW.find(([v])=>v===view)!);
 return <section className="step-content capture-page">
  <div className="section-heading"><div><span className="eyebrow">01 / Product photos</span><h2>Use the photos you already have</h2><p>Start with a front photo. Add side photos for arms. Angled and inner photos can supply further evidence when they align with approved geometry; the generated review reports which photos contributed. Review suggested shapes when automatic detection is uncertain.</p></div></div>
  <PhotoGuide onChoose={view=>{setRole('fitting');choose(view);}} disabled={busy}/>
  <label className="field">Frame construction<select aria-label="Frame construction" value={project.category} disabled={busy} onChange={e=>changeCategory(e.target.value as Category)}>{(['opaque-full-rim','thin-metal','transparent-full-rim','semi-rimless','rimless','sports-shield'] as Category[]).map(c=><option value={c} key={c}>{c.replaceAll('-',' ')}</option>)}</select></label><p className="inline-warning">This is the current project setting. Front-photo analysis can suggest a construction below; review or override it before approving the draft. Photos do not replace product dimensions.</p>
  <div className="capture-guide"><div><span className="guide-number">1</span><div><strong>Add existing photos</strong><p>No twelve-photo checklist. Keep the product fully visible.</p></div></div><div><span className="guide-number">2</span><div><strong>Check the detected shapes</strong><p>Approve the suggested boundaries or request a correction.</p></div></div><div><span className="guide-number">3</span><div><strong>Preview your draft</strong><p>Use product dimensions for scale. Inspect before trying it on.</p></div></div></div>
  <button className="button quiet" aria-expanded={advanced} onClick={()=>{setAdvanced(!advanced);setRole('fitting');}}>{advanced?'Hide advanced capture':'Advanced: extra views & validation'}</button>
  {advanced&&<div className="role-row"><label htmlFor="upload-role">New photo role</label><select id="upload-role" value={role} onChange={e=>setRole(e.target.value as 'fitting'|'validation')}><option value="fitting">Available for fitting / review</option><option value="validation">Validation · held out</option></select><span>JPEG/PNG/WebP · up to 20 MB · 40 photos per project</span></div>}
  <input ref={input} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" onChange={e=>upload(e.target.files?.[0])}/>
  <div className="subheading"><h3>{advanced?'Optional capture library':'Start here'}</h3><span>Latest fitting upload per view is used; older photos are kept.</span></div>
  <div className="capture-grid">{visible.map(([view,label,hint],index)=>{
   const photos=photosFor(view);return <div className={`capture-slot ${photos.length?'has-photo':''}`} key={view}>
    <div className="slot-top"><span className="slot-index">{String(index+1).padStart(2,'0')}</span><span className={`slot-state ${photos.length?'ready':''}`}>{photos.length?'Uploaded':view==='front'?'Required':view==='left'||view==='front-left'?'Helpful':'Optional'}</span></div>
    {photos.length?<div className="slot-photo" style={{backgroundImage:`url("${photos[0].url}")`}}><button className="slot-replace" onClick={()=>choose(view)} disabled={busy}>Upload newer photo</button></div>:<button className="slot-placeholder" onClick={()=>choose(view)} disabled={busy} aria-label={`Upload ${label}`}><span>＋</span><small>Add photo</small></button>}
    <strong>{label}</strong><p>{hint}</p>{VIEW_USE[view]&&<p className="slot-evidence-use">{VIEW_USE[view]}</p>}
    {photos.map(photo=><div className="photo-line" key={photo.id}><span className={photo.role==='validation'?'role-validation':'role-fitting'}>{photo.role==='validation'?'Held out':photo.id===photos.find(p=>p.role==='fitting')?.id?'Current':'Previous'}</span><button onClick={()=>onEditPhoto(photo.id)}>Advanced correction</button><button className="icon-button" onClick={()=>remove(photo)} title="Remove photo" aria-label={`Remove ${label} photo`}>×</button></div>)}
    {photos.map(photo=><label className="photo-view-select" key={`view-${photo.id}`}>Photo view<select aria-label={`Photo view ${photo.id.slice(0,6)}`} value={photo.view} disabled={busy} onChange={e=>relabel(photo,e.target.value)}>{[...OVERVIEW,...DETAIL].map(([v,name])=><option key={v} value={v}>{name}</option>)}</select></label>)}
    {photos.map(photo=><p className="slot-photo-quality" key={`quality-${photo.id}`}>{photo.width} × {photo.height} px{Math.min(photo.width,photo.height)<500?' · low resolution; analysis may enlarge the image without restoring detail':' · original resolution retained'}</p>)}
    {photos.flatMap(p=>p.warnings).map((warning,i)=><p className="inline-warning" key={i}>⚠ {warning}</p>)}
   </div>;
  })}</div>
  <QuickDraft key={`${project.id}:${project.revision}`} project={project} onUpdate={onUpdate} onError={onError} onEditPhoto={onEditPhoto} onGenerate={onGenerate} onDimensions={onDimensions} autoAnalyze={autoAnalyze} onAnalysisStarted={onAnalysisStarted}/>
  <p>Left and Right side views build separate arms after review. Inner / rear may add registered back-surface evidence when the geometry supports it. Use Inner left or Inner right in Advanced for a close side-on view of an individual arm’s inner face.</p>
  {armPhotos(project).map(photo=><SideDraft key={`${photo.id}:${project.revision}`} project={project} photo={photo} onUpdate={onUpdate} onError={onError} onGenerate={onGenerate} onEditPhoto={onEditPhoto} autoAnalyze={sideRequest===photo.id} onStarted={onSideStarted}/>)}
  {advanced&&<>
  <div className="subheading detail-head"><h3>Details & scale</h3><span>Optional, but useful for fine geometry</span></div>
  <div className="detail-upload"><select aria-label="Detail view" value={detailView} onChange={e=>setDetailView(e.target.value)}>{DETAIL.map(([view,label])=><option value={view} key={view}>{label}</option>)}</select><button className="button secondary" onClick={()=>choose(detailView)} disabled={busy}>{busy?'Uploading…':'Add detail photo'}</button></div>
  {project.photos.filter(p=>!OVERVIEW.some(([v])=>v===p.view)).length>0&&<div className="detail-list">{project.photos.filter(p=>!OVERVIEW.some(([v])=>v===p.view)).map(p=><div key={p.id}><img src={p.url} alt={`${p.view} detail`}/><span>{p.view}<small>{p.role}</small></span><button onClick={()=>onEditPhoto(p.id)}>Edit outline</button><button className="icon-button" onClick={()=>remove(p)} aria-label={`Remove ${p.view}`}>×</button></div>)}</div>}
  </>}
 </section>;
}
