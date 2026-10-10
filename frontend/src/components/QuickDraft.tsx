import {useEffect,useRef,useState} from 'react';
import {api} from '../api';
import type {ConstructionSuggestion,DetectionReport} from '../api';
import DetectionStatus from './DetectionStatus';
import type {Annotation,Category,Project,AnatomyReport} from '../types';
import AnatomySummary from './AnatomySummary';
import {approximateMeasurements} from '../draftMeasurements';

const CATEGORIES:Category[]=['opaque-full-rim','thin-metal','transparent-full-rim','semi-rimless','rimless','sports-shield'];
function categoryLabel(category:Category){return category==='sports-shield'?'Sports / shield':category.replaceAll('-',' ');}
type Props={project:Project;onUpdate:(p:Project)=>void;onError:(s:string)=>void;onEditPhoto:(id:string)=>void;onGenerate:(p:Project)=>void;onDimensions:()=>void;autoAnalyze:boolean;onAnalysisStarted:()=>void};
export default function QuickDraft({project,onUpdate,onError,onEditPhoto,onGenerate,onDimensions,autoAnalyze,onAnalysisStarted}:Props){
 const panel=useRef<HTMLDivElement>(null);
 const consumed=useRef(false);
 const front=project.photos.filter(p=>p.view==='front'&&p.role==='fitting').at(-1);
 const analysisToken=useRef(0);
 const contextKey=`${project.id}:${project.revision}:${front?.id??''}`;
 const latestContext=useRef(contextKey);
 latestContext.current=contextKey;
 const reviewed=front?project.annotations.filter(a=>a.photo_id===front.id&&a.approved):[];
 const [draft,setDraft]=useState<Annotation[]|null>(null);
 const [warnings,setWarnings]=useState<string[]>([]);
 const [detection,setDetection]=useState<DetectionReport|null>(null);
 const [anatomy,setAnatomy]=useState<AnatomyReport|null>(null);
 const [construction,setConstruction]=useState<ConstructionSuggestion|null>(null);
 const [selectedCategory,setSelectedCategory]=useState<Category|null>(null);
 const [manualCategory,setManualCategory]=useState(false);
 const [busy,setBusy]=useState(false);
 const [approximate,setApproximate]=useState(false);
 const [analysisBackend,setAnalysisBackend]=useState<'heuristics'|'cpu-metal'|'hybrid'|'sam2'|'sam3'>('cpu-metal');
 useEffect(()=>{analysisToken.current++;setDraft(null);setWarnings([]);setAnatomy(null);setDetection(null);setConstruction(null);setSelectedCategory(null);setManualCategory(false);setBusy(false);},[project.id,project.revision,front?.id]);
 const required=selectedCategory==='rimless'?['lens-left','lens-right']:['front','lens-left','lens-right'];
 const complete=!!draft&&!anatomy?.blocking&&!!selectedCategory&&(selectedCategory==='sports-shield'&&draft.some(a=>a.part==='shield'&&a.points.length>=3)||required.every(part=>draft.some(a=>a.part===part&&a.points.length>=3)));
 const missing=['frame_width','lens_width','lens_height','bridge_width','temple_length'].filter(k=>!(project.measurements[k]?.value>0));
 async function analyze(){
  if(!front)return;
  const requestToken=++analysisToken.current;
  const requestContext=contextKey;
  const priorManual=manualCategory&&!!construction,priorCategory=selectedCategory;
  setBusy(true);onError('');setDraft(null);setAnatomy(null);setDetection(null);setConstruction(null);
  try{
   const result=await api.extract(project.id,front.id,analysisBackend||'cpu-metal');
   if(requestToken!==analysisToken.current||requestContext!==latestContext.current)return;
   setDraft(result.annotations.map(a=>({...a,approved:false})));
   setWarnings(result.warnings);
   setDetection(result.detection_report??null);
   setAnatomy(result.anatomy??null);
   setConstruction(result.construction);
   const suggested=result.construction?.category;
   const useSuggestion=!!suggested&&result.construction.confidence!=='low';
   setSelectedCategory(priorManual&&priorCategory?priorCategory:useSuggestion?suggested:null);
   setManualCategory((priorManual&&!!priorCategory)||!useSuggestion);
  }catch(e){if(requestToken===analysisToken.current&&requestContext===latestContext.current){const message=e instanceof Error?e.message:String(e);setWarnings([message]);onError(message);}}finally{if(requestToken===analysisToken.current&&requestContext===latestContext.current)setBusy(false);}
 }
 function showReviewed(){analysisToken.current++;setDraft(reviewed);setAnatomy(null);setDetection(null);setWarnings(['Showing the approved outlines saved for this front photo.']);setConstruction(null);setSelectedCategory(project.category);setManualCategory(true);}
 useEffect(()=>{
  if(!autoAnalyze){consumed.current=false;return;}
  if(consumed.current)return;
  consumed.current=true;
  panel.current?.scrollIntoView({behavior:'smooth',block:'start'});
  if(!busy)void analyze();
  onAnalysisStarted();
 },[autoAnalyze]);
 async function approve(){
  if(!front||!draft||!complete||!selectedCategory||(missing.length&&!approximate))return;setBusy(true);onError('');
  try{
   const audit=await api.auditAnatomy(project.id,front.id,draft,selectedCategory);setAnatomy(audit);
   if(audit.blocking)throw new Error('The structure audit found conflicting geometry. Correct it in Advanced before approval.');
   const annotations=[...project.annotations.filter(a=>a.photo_id!==front.id),...draft.map(a=>({...a,approved:true,source:a.approved?a.source:`merchant-reviewed:${a.source}`}))];
   const measurements=approximate?approximateMeasurements(draft,project.measurements):project.measurements;
   const fresh=await api.patch(project.id,{annotations,measurements,category:selectedCategory});onUpdate(fresh);onGenerate(fresh);
  }catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 async function correct(){
  if(!front||busy)return;
  // Preserve detected suggestions for the editor instead of making the merchant
  // start over. This does not approve automatic geometry or start generation.
  setBusy(true);onError('');
  try{
   if(draft?.length){
    const annotations=[...project.annotations.filter(a=>a.photo_id!==front.id),...draft];
    onUpdate(await api.patch(project.id,{annotations}));
   }
   onEditPhoto(front.id);
  }catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 return <div ref={panel} className="quick-draft review-card">
  <h3>Make a draft from your photos</h3>
  <p>Start with a straight front photo. We suggest its frame and lens boundaries; check the highlighted shapes, then approve once. No tracing is needed when detection works.</p>
  <p>A front-only draft has no verified arms. Add a side photo and use its arm-analysis panel below; Advanced is available for corrections. Hidden geometry and material depth remain approximate without additional evidence.</p>
  {front&&<label className="field">Front photo analysis<select aria-label="Front photo analysis" value={analysisBackend||'cpu-metal'} disabled={busy} onChange={e=>{setAnalysisBackend(e.target.value as 'heuristics'|'cpu-metal'|'hybrid'|'sam2'|'sam3');analysisToken.current++;setDraft(null);setDetection(null);setAnatomy(null);setWarnings([]);setConstruction(null);setSelectedCategory(null);setManualCategory(false);}}><option value="cpu-metal">CPU · thin metal + contours (no AI)</option><option value="sam3">SAM 3 · CPU</option><option value="sam2">SAM 2.1 · AI assisted</option><option value="hybrid">Legacy lens model · experimental</option><option value="heuristics">Standard analysis</option></select></label>}
  {front&&analysisBackend==='sam3'&&<p>SAM 3 analyzes this photo locally on CPU using eyewear text prompts. The official checkpoint must be installed first. Analysis can take several minutes; outlines still require source checks and review.</p>}
  {front&&analysisBackend==='sam2'&&<p>SAM 2.1 uses automatic image prompts with a coarse lens locator. Analysis can take longer on CPU. Component suggestions remain experimental; a general segmentation mask does not identify hidden hinge construction.</p>}
  {!front?<div className="notice warning">Add a straight front photo to start. You do not need all twelve views.</div>:<div className="header-actions"><button className="button primary" disabled={busy} onClick={analyze}>{busy?'Analyzing…':'Analyze current front photo'}</button>{reviewed.length>0&&<button className="button secondary" disabled={busy} onClick={showReviewed}>Show approved outlines</button>}</div>}
  {!draft&&warnings.map((w,i)=><p className="inline-warning" role="alert" key={i}>{w}</p>)}
  {draft&&front&&<>
   <svg className="draft-overlay" viewBox={`0 0 ${front.width} ${front.height}`} role="img" aria-label="Detected outlines on current front photo">
    <image href={front.url} width={front.width} height={front.height}/>
    {draft.map((a,i)=><polygon key={i} points={a.points.map(p=>`${p.x},${p.y}`).join(' ')} fill="none" stroke={a.part.startsWith('hinge-')?'#916ae0':a.part==='front-support'?'#b58116':a.part.startsWith('nose-pad-')?'#168bdb':a.part==='front'?'#16a36e':a.part==='bridge-opening'?'#d64bc7':'#e66b16'} strokeWidth={Math.max(front.width/350,1)} />)}
   </svg>
   <p>Green: outer frame. Gold: separate front support. Purple: visible hinge hardware. Orange: lens boundaries and separate rimless bridge hardware. Pink: extra bridge opening. Blue: visible nose-pad silhouettes, when detected. Rimless lenses do not require an outer frame. Check that these match your product, not shadows or folded arms. Hidden edges and component depth remain inferred and need your review.</p>
   {warnings.map((w,i)=><p className="inline-warning" key={i}>{w}</p>)}
   <DetectionStatus report={detection}/>
   <AnatomySummary report={anatomy}/>
   <div className="construction-review">
    <strong>Frame construction</strong>
    {construction?<p>{construction.category?<>Photo suggestion: <strong>{categoryLabel(construction.category)}</strong> ({construction.confidence} confidence).</>:<>The photo did not identify a frame construction.</>} {construction.reason}</p>:<p>Using the frame construction already saved with this project.</p>}
    {construction?.category&&construction.confidence==='low'&&<p className="inline-warning">This suggestion is uncertain. Choose the construction before approving.</p>}
    {construction?.category&&construction.confidence!=='low'&&<label className="construction-mode"><input type="radio" name="draft-construction-mode" checked={!manualCategory} onChange={()=>{setManualCategory(false);setSelectedCategory(construction.category);}}/> Use photo suggestion</label>}
    <label className="construction-mode"><input type="radio" name="draft-construction-mode" checked={manualCategory} onChange={()=>{setManualCategory(true);setSelectedCategory(project.category);}}/> Choose manually</label>
    <label className="field">Construction to use for this draft
     <select aria-label="Construction to use for this draft" value={selectedCategory??''} disabled={!manualCategory} onChange={e=>setSelectedCategory(e.target.value?e.target.value as Category:null)}>
      <option value="" disabled>Choose a frame construction</option>
      {CATEGORIES.map(category=><option value={category} key={category}>{categoryLabel(category)}{category===project.category?' · current project':''}</option>)}
     </select>
    </label>
    {selectedCategory&&<p>{manualCategory?'Manual choice':'Photo suggestion selected'}. {selectedCategory!==project.category?<>Approval will change the project from {categoryLabel(project.category)} to <strong>{categoryLabel(selectedCategory)}</strong>.</>:<>This matches the current project setting.</>}</p>}
   </div>
   {!selectedCategory&&<div className="notice warning" role="status">Choose a frame construction to check the required outlines and enable approval.</div>}
   {selectedCategory&&!complete&&<div className="notice warning" role="status"><strong>Automatic detection needs help with this photo</strong><p>Not confidently traced: {required.filter(part=>!draft.some(a=>a.part===part&&a.points.length>=3)).map(part=>({'front':'outer frame','lens-left':'left lens opening','lens-right':'right lens opening'}[part]??part)).join(', ')||'required geometry'}.</p><p>This can happen even with a clear photo, especially with transparent rims, reflections or hidden edges. Any usable suggestions are shown above. Check the upload examples at the top, try another front photo if available, or correct the uncertain boundaries in Advanced. We will not substitute an unrelated frame.</p></div>}
   {missing.length>0&&<div className="notice warning"><p>No physical measurements? You can make a visual draft. Missing overall width and arm length use a placeholder of 140 mm; front proportions are estimated from the outlines where possible. These are not your product’s measured dimensions and cannot establish fit.</p><label><input type="checkbox" checked={approximate} onChange={e=>setApproximate(e.target.checked)}/> Use approximate size for this draft — not accurate fit</label><p>Or <button onClick={onDimensions}>enter product dimensions →</button></p></div>}
   <div className="header-actions"><button className="button primary" disabled={busy||!complete||(!!missing.length&&!approximate)} onClick={approve}>Approve outlines & generate draft</button><button className="button secondary" disabled={busy} onClick={correct}>Correct in Advanced</button></div>
  </>}
 </div>;
}
