import {useEffect,useRef,useState} from 'react';
import {api} from '../api';
import type {DetectionReport} from '../api';
import DetectionStatus from './DetectionStatus';
import type {Annotation,Photo,Project,AnatomyReport} from '../types';
import AnatomySummary from './AnatomySummary';
type Props={project:Project;photo:Photo;onUpdate:(p:Project)=>void;onError:(s:string)=>void;onGenerate:(p:Project)=>void;onEditPhoto:(id:string)=>void;autoAnalyze:boolean;onStarted:()=>void};
export default function SideDraft({project,photo,onUpdate,onError,onGenerate,onEditPhoto,autoAnalyze,onStarted}:Props){
 const panel=useRef<HTMLElement>(null);const consumed=useRef(false);
 const context=useRef('');context.current=`${project.id}:${photo.id}:${project.revision}`;
 const [annotation,setAnnotation]=useState<Annotation|null>(null);
 const [warnings,setWarnings]=useState<string[]>([]);
 const [detection,setDetection]=useState<DetectionReport|null>(null);
 const [anatomy,setAnatomy]=useState<AnatomyReport|null>(null);
 const [appearance,setAppearance]=useState<NonNullable<Awaited<ReturnType<typeof api.extract>>['appearance_evidence']>>([]);
 const [analysisBackend,setAnalysisBackend]=useState<'heuristics'|'sam3'|'sam2'>('heuristics');
 const [busy,setBusy]=useState(false);
 const oppositePart=photo.view.includes('left')?'temple-right':'temple-left';
 const savedMirror=project.annotations.some(a=>a.photo_id===photo.id&&a.part===oppositePart&&a.approved&&a.source.startsWith('explicit-mirrored-side-assumption:'));
 const [mirror,setMirror]=useState(savedMirror);
 useEffect(()=>{if(!autoAnalyze){consumed.current=false;return;}if(consumed.current)return;consumed.current=true;panel.current?.scrollIntoView({behavior:'smooth',block:'start'});if(!busy)void analyze();onStarted();},[autoAnalyze]);
 useEffect(()=>{setAppearance([]);setAnnotation(null);setWarnings([]);setAnatomy(null);setDetection(null);setBusy(false);},[project.id,photo.id,project.revision]);
 async function analyze(){const requestContext=context.current;setBusy(true);setAppearance([]);setAnatomy(null);setDetection(null);onError('');try{const r=await api.extract(project.id,photo.id,analysisBackend);if(context.current!==requestContext)return;const arm=r.annotations.find(a=>a.part.startsWith('temple-'));setAnnotation(arm?{...arm,approved:false}:null);setWarnings(r.warnings);setDetection(r.detection_report??null);setAnatomy(r.anatomy??null);setAppearance(r.appearance_evidence??[]);}catch(e){if(context.current===requestContext)onError(e instanceof Error?e.message:String(e));}finally{if(context.current===requestContext)setBusy(false);}}
 async function approve(){if(!annotation)return;setBusy(true);try{
  const audit=await api.auditAnatomy(project.id,photo.id,[annotation],project.category);setAnatomy(audit);
  if(audit.blocking)throw new Error('Correct the arm outline and hinge/tip anchors before approval.');
  const opposite=annotation.part==='temple-left'?'temple-right':'temple-left';
  const oppositeExists=project.annotations.some(a=>a.part===opposite&&a.approved&&!a.source.includes('mirrored'));
  if(mirror&&oppositeExists)throw new Error('The opposite arm already has reviewed geometry. Turn off mirroring to preserve it.');
  const oldMirrored=project.annotations.some(a=>a.part===opposite&&a.photo_id===photo.id&&a.source.includes('mirrored'));
  const replaced=[annotation.part,...(mirror||oldMirrored?[opposite]:[])];
  const annotations=project.annotations.filter(a=>!replaced.includes(a.part));
  annotations.push({...annotation,approved:true,source:`merchant-reviewed-side:${annotation.source}`});
  if(mirror)annotations.push({...annotation,part:opposite,approved:true,source:`explicit-mirrored-side-assumption:${annotation.source}`});
  const fresh=await api.patch(project.id,{annotations});onUpdate(fresh);onGenerate(fresh);
 }catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 async function keepReference(){setBusy(true);try{const fresh=await api.relabelPhoto(project.id,photo.id,'detail');onUpdate(fresh);onGenerate(fresh);}catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 async function adjust(){if(busy)return;setBusy(true);onError('');try{
  if(annotation){
   const annotations=[...project.annotations.filter(a=>a.photo_id!==photo.id||a.part!==annotation.part),{...annotation,approved:false}];
   onUpdate(await api.patch(project.id,{annotations}));
  }
  onEditPhoto(photo.id);
 }catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 return <section ref={panel} className="quick-draft review-card"><h3>Reconstruct the {photo.view} arm</h3><p>Separate one arm from this side photo, then review the hinge and tip. No generic arm is added.</p><label className="field">Side photo analysis<select aria-label="Side photo analysis" value={analysisBackend} disabled={busy} onChange={e=>{setAnalysisBackend(e.target.value as 'heuristics'|'sam3'|'sam2');setAnnotation(null);setDetection(null);setAnatomy(null);setWarnings([]);}}><option value="heuristics">CPU · source contours (no AI)</option><option value="sam3">SAM 3 · CPU</option><option value="sam2">SAM 2.1 · AI assisted</option></select></label><button className="button secondary" disabled={busy} onClick={analyze}>{busy?'Working…':'Analyze side arm'}</button>
 <svg className="draft-overlay" viewBox={`0 0 ${photo.width} ${photo.height}`} role="img" aria-label={`${photo.view} arm preview`}><image href={photo.url} width={photo.width} height={photo.height}/>{annotation&&<><polygon points={annotation.points.map(p=>`${p.x},${p.y}`).join(' ')} fill="#ff8c0020" stroke="#e66b16" strokeWidth={photo.width/250}/>{Object.entries(annotation.anchors).map(([name,p])=><g key={name}><circle cx={p.x} cy={p.y} r={photo.width/70} fill={name==='hinge'?'#00a86b':'#d54cdf'}/><text x={p.x} y={p.y-photo.width/40} fontSize={photo.width/22}>{name}</text></g>)}</>}</svg>
 {warnings.map((w,i)=><p className="inline-warning" key={i}>{w}</p>)}
 <DetectionStatus report={detection}/>
 <AnatomySummary report={anatomy}/>
 {appearance.map((e,i)=>{const xs=e.points.map(p=>p.x),ys=e.points.map(p=>p.y);const x=Math.min(...xs)-8,y=Math.min(...ys)-8,w=Math.max(...xs)-x+8,h=Math.max(...ys)-y+8;return <div className="review-card" key={i}><h4>Opposite arm: visible inner-surface candidate</h4><svg viewBox={`${x} ${y} ${w} ${h}`} style={{width:'100%',maxHeight:220}} role="img" aria-label="Opposite arm appearance evidence"><image href={photo.url} width={photo.width} height={photo.height}/><polygon points={e.points.map(p=>`${p.x},${p.y}`).join(' ')} fill="none" stroke="#149ca8" strokeWidth={photo.width/500}/></svg><p>{e.possible_mark_regions>0?'Possible printed or reflective details detected. ':''}{e.reason}</p><p>Excluded from selected-arm geometry. Generation applies only the visible tip region to the inner surface of {e.part.replaceAll('-',' ')} when that arm has approved geometry. Tip/width registration is approximate; the outer surface remains unchanged.</p></div>;})}
 <button className="button quiet" disabled={busy} onClick={keepReference}>Use this photo as reference only; omit this unverified arm</button>
 {annotation&&<><label><input type="checkbox" checked={mirror} onChange={e=>setMirror(e.target.checked)}/> Also mirror this reviewed shape for the opposite arm (assume symmetry)</label><p>This is an explicit approximation, not evidence of the unseen arm. Leave unchecked to omit it.{savedMirror?' Your previously approved mirroring choice is retained for this photo.':''}</p></>}
 <div className="header-actions"><button className="button primary" disabled={busy||!annotation} onClick={approve}>Approve arm & regenerate</button><button className="button quiet" disabled={busy} onClick={adjust}>Adjust outline / hinge / tip</button></div></section>;
}
