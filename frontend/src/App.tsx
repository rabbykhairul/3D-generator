import {useCallback,useEffect,useState} from 'react';
import {api} from './api';
import type {Category,Job,Project} from './types';
import Capture from './components/Capture';
import OutlineEditor from './components/OutlineEditor';
import Measurements from './components/Measurements';
import ModelViewer from './components/ModelViewer';
import EvidenceSummary from './components/EvidenceSummary';
import JointFitExperiment from './components/JointFitExperiment';
import {armPhotos} from './photoEvidence';
import {measurementIssue} from './draftMeasurements';

const CATEGORIES:{value:Category;label:string}[]=[
 {value:'opaque-full-rim',label:'Opaque full-rim'},
 {value:'thin-metal',label:'Thin metal'},
 {value:'transparent-full-rim',label:'Transparent full-rim'},
 {value:'semi-rimless',label:'Semi-rimless'},
 {value:'rimless',label:'Rimless'},
 {value:'sports-shield',label:'Sports / shield'}
];
type Step='capture'|'outlines'|'dimensions'|'review';
const STEPS:{id:Step;label:string;subtitle:string}[]=[{id:'capture',label:'Photos',subtitle:'Upload & prepare draft'},{id:'dimensions',label:'Dimensions',subtitle:'Product specifications'},{id:'review',label:'Review / AR',subtitle:'Inspect reconstruction'}];
type Sample={id:string;name:string;category:string;kind:string;description:string};

function errorText(e:unknown){return e instanceof Error?e.message:String(e);}
function displayCategory(category:string){return CATEGORIES.find(c=>c.value===category)?.label||category;}

export default function App(){
 const [projects,setProjects]=useState<Project[]>([]);
 const [active,setActive]=useState<Project|null>(null);
 const [step,setStep]=useState<Step>('capture');
 const [viewerMode,setViewerMode]=useState<'inspect'|'ar'>('inspect');
 const [job,setJob]=useState<Job|null>(null);
 const [error,setError]=useState('');
 const [loading,setLoading]=useState(true);
 const [busy,setBusy]=useState(false);
 const [showCreate,setShowCreate]=useState(false);
 const [showSamples,setShowSamples]=useState(false);
 const [samples,setSamples]=useState<Sample[]>([]);
 const [name,setName]=useState('');
 const [category,setCategory]=useState<Category>('opaque-full-rim');
 const [editorPhotoId,setEditorPhotoId]=useState<string|undefined>();
 const [sidebarOpen,setSidebarOpen]=useState(false);
 const [analysisProject,setAnalysisProject]=useState<string|null>(null);
 const [sideRequest,setSideRequest]=useState<string|null>(null);
 const [removeTarget,setRemoveTarget]=useState<Project|null>(null);
 const [trash,setTrash]=useState<Project[]|null>(null);
 async function removeSavedProject(){
  if(!removeTarget)return;setBusy(true);setError('');
  try{await api.removeProject(removeTarget.id);const remaining=await api.list();setProjects(remaining);if(active?.id===removeTarget.id){setJob(null);setActive(remaining[0]?await api.get(remaining[0].id):null);setStep('capture');}setRemoveTarget(null);}catch(e){setError(errorText(e));}finally{setBusy(false);}
 }
 async function openTrash(){try{setTrash(await api.archived());}catch(e){setError(errorText(e));}}
 async function restoreSavedProject(id:string){setBusy(true);try{await api.restoreProject(id);setProjects(await api.list());setTrash(await api.archived());}catch(e){setError(errorText(e));}finally{setBusy(false);}}
 const update=useCallback((project:Project)=>{setActive(project);setProjects(p=>{const exists=p.some(x=>x.id===project.id);return exists?p.map(x=>x.id===project.id?project:x):[project,...p];});},[]);
 useEffect(()=>{let alive=true;api.list().then(items=>{if(!alive)return;setProjects(items);if(items.length)api.get(items[0].id).then(p=>{if(alive)setActive(p);}).catch(e=>{if(alive)setError(errorText(e));});}).catch(e=>{if(alive)setError(errorText(e));}).finally(()=>{if(alive)setLoading(false);});return()=>{alive=false;};},[]);
 useEffect(()=>{
  if(!job||!['queued','processing'].includes(job.status))return;
  let alive=true;let inFlight=false;
  const timer=window.setInterval(async()=>{
   if(inFlight)return;inFlight=true;
   try{
    const next=await api.job(job.id);
    if(!alive)return;
    if(!['queued','processing'].includes(next.status)){
     const fresh=await api.get(next.project_id);
     if(!alive)return;
     update(fresh);
    }
    if(alive)setJob(next);
   }catch(e){if(alive)setError(errorText(e));}finally{inFlight=false;}
  },750);
  return()=>{alive=false;window.clearInterval(timer);};
 },[job?.id,job?.status,update]);
 async function selectProject(project:Project){setBusy(true);setError('');try{const fresh=await api.get(project.id);setActive(fresh);setJob(null);setStep('capture');setEditorPhotoId(undefined);setViewerMode('inspect');setSidebarOpen(false);}catch(e){setError(errorText(e));}finally{setBusy(false);}}
 async function create(){if(!name.trim()){setError('Enter a project name.');return;}setBusy(true);setError('');try{const project=await api.create(name.trim(),category);update(project);setJob(null);setStep('capture');setShowCreate(false);setName('');setSidebarOpen(false);}catch(e){setError(errorText(e));}finally{setBusy(false);}}
 async function openSamples(){setShowSamples(true);setBusy(true);setError('');try{setSamples(await api.samples());}catch(e){setError(errorText(e));}finally{setBusy(false);}}
 async function importSample(id:string){setBusy(true);setError('');try{const project=await api.importSample(id);update(project);setJob(null);setStep('capture');setShowSamples(false);setSidebarOpen(false);}catch(e){setError(errorText(e));}finally{setBusy(false);}}
 async function generateApproved(project:Project){
  const issue=measurementIssue(project.measurements);if(issue){setError(issue);setStep('dimensions');return;}
  const pending=armPhotos(project).find(photo=>!project.annotations.some(a=>a.photo_id===photo.id&&a.part===`temple-${photo.view.includes('left')?'left':'right'}`&&a.approved&&a.source!=='merchant-reviewed-side'&&!a.source.includes('single-visible-arm-suggestion')));
  if(pending){setError('');setStep('capture');setSideRequest(pending.id);return;}
  setBusy(true);setError('');setStep('review');try{setJob(await api.generate(project.id));}catch(e){setError(errorText(e));}finally{setBusy(false);}
 }
 async function generate(){
  if(!active)return;
  const issue=measurementIssue(active.measurements);if(issue){setError(issue);setStep('dimensions');return;}
  const front=active.photos.filter(p=>p.view==='front'&&p.role==='fitting').at(-1);
  if(!front){setStep('capture');setError('Add a straight front photo in the Front slot to start.');return;}
  const approved=active.annotations.filter(a=>a.photo_id===front.id&&a.approved).map(a=>a.part);
  const required=active.category==='rimless'?['lens-left','lens-right']:['front','lens-left','lens-right'];
  const ready=required.every(p=>approved.includes(p))||(active.category==='sports-shield'&&approved.includes('shield'));
  const scaled=['frame_width','lens_width','lens_height','bridge_width','temple_length'].every(k=>active.measurements[k]?.value>0);
  const legacyOutline=active.annotations.some(a=>a.photo_id===front.id&&['automatic','merchant-reviewed:automatic','automatic-shadow-aware-v3','merchant-reviewed:automatic-shadow-aware-v3'].includes(a.source));
  if(!ready||!scaled||legacyOutline){setError('');setStep('capture');setAnalysisProject(active.id);return;}
  await generateApproved(active);
 }
 function editPhoto(photoId:string){setEditorPhotoId(photoId);setStep('outlines');}
 const current=active;
 const artifact=current?.artifact;
 const outdated=!!artifact&&(current?.stale||artifact.revision!==current.revision);
 const approved=current?.annotations.filter(a=>a.approved).length||0;
 const measured=['frame_width','lens_width','lens_height','bridge_width','temple_length'].filter(k=>(current?.measurements[k]?.value??0)>0).length;
 const correctionText=`${job?.error||''} ${JSON.stringify(job?.details||{})}`.toLowerCase();
 const correctionStep:Step=/measure|width|height|scale|bridge|temple length/.test(correctionText)?'dimensions':/outline|contour|polygon|lens|geometry/.test(correctionText)?'outlines':/photo|view|image/.test(correctionText)?'capture':'outlines';
 const correctionLabel=correctionStep==='dimensions'?'Edit dimensions':correctionStep==='capture'?'Review photos':'Open editor';
 return <div className="app-shell">
  <aside className={`sidebar ${sidebarOpen?'open':''}`}><div className="brand"><div className="brand-mark"><span>F</span></div><div><strong>FRAME / LAB</strong><small>Reconstruction studio</small></div><button className="mobile-close" onClick={()=>setSidebarOpen(false)} aria-label="Close navigation">×</button></div>
   <div className="workspace-label">WORKSPACE</div><div className="sidebar-actions"><button className="new-project" onClick={()=>setShowCreate(true)}>＋ <span>New project</span></button><button className="sample-button" onClick={openSamples}>↗ <span>Import sample</span></button></div>
   <div className="project-list-head"><span>PROJECTS</span><span>{projects.length}</span></div><nav className="project-list" aria-label="Projects">{projects.map(p=><div className="saved-project-row" key={p.id}><button className={`project-nav ${current?.id===p.id?'active':''}`} onClick={()=>selectProject(p)} disabled={busy}><span className="project-glyph">⌁</span><span className="project-nav-text"><strong>{p.name}</strong><small>{displayCategory(p.category)} · {p.id.slice(0,6)}</small></span>{p.stale&&<span className="stale-dot" title="Model needs regeneration"/>}</button><button className="remove-project" aria-label={`Remove ${p.name} ${p.id.slice(0,6)}`} title="Remove saved project" disabled={busy} onClick={()=>setRemoveTarget(p)}>×</button></div>)}{!projects.length&&!loading&&<p className="sidebar-empty">Create a project or import a sample to start.</p>}</nav>
   <button className="sample-button" onClick={openTrash}>Removed projects / Restore</button>
   <div className="sidebar-bottom"><span className="status-light"/> Local research workspace <small>Photos and models stay on this machine.</small></div>
  </aside>
  <main className="main-area"><header className="topbar"><button className="mobile-menu" onClick={()=>setSidebarOpen(true)} aria-label="Open navigation">☰</button><div className="breadcrumb"><span>Workspace</span><b>/</b><strong>{current?.name||'Projects'}</strong></div><div className="topbar-right"><span className="local-badge"><i/> LOCAL STUDIO</span>{current&&<span className="revision-badge">Revision {current.revision}</span>}</div></header>
   {error&&<div className="global-error" role="alert"><span>⚠</span><p>{error}</p><button onClick={()=>setError('')} aria-label="Dismiss error">×</button></div>}
   {loading?<div className="app-loading">Loading workspace…</div>:!current?<div className="welcome-state"><div className="welcome-icon">⌁</div><span className="eyebrow">FRAME / LAB · LOCAL RESEARCH</span><h1>Start a reconstruction</h1><p>Bring in product photos, trace geometry, set dimensions, and generate a model for inspection and webcam try-on.</p><div><button className="button primary" onClick={()=>setShowCreate(true)}>Create project</button><button className="button secondary" onClick={openSamples}>Browse samples</button></div><div className="welcome-note">Sample projects identify public source photos and synthetic fixtures. Missing dimensions or views stay marked as missing.</div></div>:<>
    <div className="project-header"><div><span className="eyebrow">PROJECT / {current.id.slice(0,8).toUpperCase()}</span><h1>{current.name}</h1><div className="project-meta"><span>{displayCategory(current.category)}</span><span>·</span><span>{current.photos.length} photos</span><span>·</span><span>{approved} approved traces</span></div></div><div className="header-actions"><span className={`artifact-state ${outdated?'stale':artifact?'ready':'pending'}`}>{outdated?'Model out of date':artifact?'Model ready':'No model yet'}</span><button className="button primary" onClick={generate} disabled={busy||job?.status==='queued'||job?.status==='processing'}>{busy?'Working…':outdated?'Regenerate model':'Generate model'} <span>→</span></button></div></div>
    <div className="workflow-tabs" role="tablist" aria-label="Project workflow">{STEPS.map((item,i)=><button role="tab" aria-selected={step===item.id} className={`workflow-tab ${step===item.id?'active':''}`} key={item.id} onClick={()=>setStep(item.id)}><span className="tab-number">{String(i+1).padStart(2,'0')}</span><span><strong>{item.label}</strong><small>{item.subtitle}</small></span></button>)}</div>
    <button className="button quiet" onClick={()=>setStep(step==='outlines'?'capture':'outlines')}>{step==='outlines'?'Back to photos':'Advanced outline editor'}</button>
    {artifact&&!outdated&&artifact.metadata.reconstruction_status==='incomplete'&&<div className="notice warning" role="status">Incomplete reconstruction: unverified arms are omitted. This front-only preview is not a complete try-on asset. Add reviewed side outlines to reconstruct the missing parts.</div>}
    {['frame_width','lens_width','lens_height','bridge_width','temple_length'].some(k=>['default','inferred'].includes(current.measurements[k]?.source))&&<div className="notice warning" role="status">Approximate-size draft — some dimensions are assumed or inferred, not measured. AR placement does not establish physical fit. Replace them with product specifications when available.</div>}
    {step==='capture'&&<Capture key={current.id} project={current} onUpdate={update} onError={setError} onEditPhoto={editPhoto} onGenerate={generateApproved} onDimensions={()=>setStep('dimensions')} autoAnalyze={analysisProject===current.id} onAnalysisStarted={()=>setAnalysisProject(null)} sideRequest={sideRequest} onSideStarted={()=>setSideRequest(null)}/>}
    {step==='outlines'&&<OutlineEditor project={current} onUpdate={update} onError={setError} initialPhotoId={editorPhotoId}/>}
    {step==='dimensions'&&<Measurements project={current} onUpdate={update} onError={setError}/>}

    <JointFitExperiment project={current} onUpdate={update} onError={setError}/>
    {step==='review'&&<section className="step-content review-page"><div className="section-heading"><div><span className="eyebrow">04 / Output</span><h2>Review & try on</h2><p>Generate from approved geometry and current measurements. Inspect the result against photos before using AR.</p></div></div>
      <div className="readiness-grid"><div><small>PHOTO EVIDENCE</small><strong>{current.photos.filter(p=>p.role==='fitting').length} fitting · {current.photos.filter(p=>p.role==='validation').length} validation</strong><button onClick={()=>setStep('capture')}>Manage photos →</button></div><div><small>GEOMETRY</small><strong>{approved} approved outlines</strong><button onClick={()=>setStep('outlines')}>Correct outlines →</button></div><div><small>SCALE</small><strong>{measured} of 5 required values</strong><button onClick={()=>setStep('dimensions')}>Edit dimensions →</button></div></div>
      {job&&<div className={`job-banner ${job.status}`} role="status"><div className="job-symbol">{job.status==='completed'?'✓':job.status==='failed'||job.status==='needs-correction'?'!':'◌'}</div><div><strong>{job.status==='needs-correction'?'Correction needed':job.status==='failed'?'Generation failed':job.status==='completed'?'Model generated':job.status==='queued'?'Queued for reconstruction':'Reconstructing geometry'}</strong><p>{job.error||job.stage||`Input revision ${job.revision}`}</p>{job.details&&<details><summary>Details</summary><pre>{JSON.stringify(job.details,null,2)}</pre></details>}</div>{job.status==='needs-correction'&&<button className="button secondary" onClick={()=>setStep(correctionStep)}>{correctionLabel} →</button>}{job.status==='failed'&&<button className="button secondary" onClick={generate}>Retry</button>}</div>}
      {artifact&&!outdated?<><div className="viewer-header"><div><h3>3D inspection</h3><span>{`Generated from revision ${artifact.revision}`}</span></div><div className="viewer-switch"><button className={viewerMode==='inspect'?'active':''} onClick={()=>setViewerMode('inspect')}>Inspect</button><button className={viewerMode==='ar'?'active':''} onClick={()=>setViewerMode('ar')}>Webcam AR</button></div></div><ModelViewer key={artifact.job_id} project={current} mode={viewerMode} onModeChange={setViewerMode}/><div className="artifact-links"><a className="button secondary" href={artifact.model_url} download>↓ Download GLB</a><a className="button quiet" href={artifact.metadata_url} download>↓ Download metadata</a></div>
       <EvidenceSummary project={current}/><div className="review-bottom"><div className="review-card"><h3>Model provenance</h3><p>Measurements are recorded with their source. Geometry and appearance warnings are stored in the metadata.</p><dl><div><dt>Frame width</dt><dd>{current.measurements.frame_width?.value||'—'} mm · {current.measurements.frame_width?.source||'missing'}</dd></div><div><dt>Revision</dt><dd>{artifact.revision}</dd></div><div><dt>Parts</dt><dd>{Array.isArray(artifact.metadata?.parts)?artifact.metadata.parts.length:'See metadata'}</dd></div></dl></div><div className="review-card"><h3>Compare with evidence</h3><p>Check silhouette, lens openings, hinges and temple shape against source photos. A visually plausible preview does not establish physical fit.</p><div className="reference-strip">{current.photos.slice(0,4).map(p=><img src={p.url} alt={`${p.view} reference`} key={p.id}/>)}</div></div></div></>:<div className="empty-panel review-empty"><span>◇</span><h3>{outdated?'Your photos or settings changed':'No model generated yet'}</h3><p>{outdated?'The previous model is hidden because it does not represent your current inputs. Analyze the latest front photo, approve its shapes and generate a new draft.':'Start with your product photos, review the detected shapes and supply product dimensions.'}</p><button className="button primary" onClick={()=>setStep('capture')}>Prepare draft from photos →</button>{outdated&&<a className="button quiet" href={artifact!.model_url} download>Download previous model (out of date)</a>}</div>}
    </section>}
   </>}
  </main>
  {removeTarget&&<div className="modal-backdrop"><div className="modal" role="dialog" aria-modal="true" aria-label="Remove saved project"><h2>Remove this saved project?</h2><p>{removeTarget.name} · {removeTarget.id.slice(0,6)}</p><p>It will disappear from the sidebar. Photos, models and history are retained; restore it from Removed projects at any time.</p><div className="modal-actions"><button className="button quiet" disabled={busy} onClick={()=>setRemoveTarget(null)}>Cancel</button><button className="button primary" disabled={busy} onClick={removeSavedProject}>Remove project</button></div></div></div>}
  {trash&&<div className="modal-backdrop"><div className="modal" role="dialog" aria-modal="true" aria-label="Removed projects"><h2>Removed projects</h2><p>Nothing here is permanently deleted.</p><div className="trash-list">{trash.map(p=><div key={p.id}><span>{p.name} · {p.id.slice(0,6)}</span><button disabled={busy} onClick={()=>restoreSavedProject(p.id)}>Restore</button></div>)}{!trash.length&&<p>No removed projects.</p>}</div><button className="button secondary" onClick={()=>setTrash(null)}>Close</button></div></div>}
  {showCreate&&<div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)setShowCreate(false);}}><div className="modal" role="dialog" aria-modal="true" aria-labelledby="create-title"><button className="modal-close" onClick={()=>setShowCreate(false)} aria-label="Close">×</button><span className="eyebrow">NEW PROJECT</span><h2 id="create-title">What are you reconstructing?</h2><p>Start with a frame category. You can add evidence as you go.</p><label className="field">Project name<input autoFocus value={name} onChange={e=>setName(e.target.value)} onKeyDown={e=>{if(e.key==='Enter')create();}} placeholder="e.g. Black acetate wayfarer" maxLength={120}/></label><label className="field">Frame category<select value={category} onChange={e=>setCategory(e.target.value as Category)}>{CATEGORIES.map(c=><option key={c.value} value={c.value}>{c.label}</option>)}</select></label><div className="modal-actions"><button className="button quiet" onClick={()=>setShowCreate(false)}>Cancel</button><button className="button primary" onClick={create} disabled={busy}>{busy?'Creating…':'Create project'}</button></div></div></div>}
  {showSamples&&<div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)setShowSamples(false);}}><div className="modal samples-modal" role="dialog" aria-modal="true" aria-labelledby="samples-title"><button className="modal-close" onClick={()=>setShowSamples(false)} aria-label="Close">×</button><span className="eyebrow">SAMPLE LIBRARY</span><h2 id="samples-title">Import a sample</h2><p>Samples identify their source and evidence gaps. Synthetic fixtures are for testing geometry, not proof of photo accuracy.</p><div className="sample-list">{samples.length?samples.map(s=><div className="sample-item" key={s.id}><div><span className={`sample-kind ${s.kind==='synthetic'?'synthetic':''}`}>{s.kind}</span><h3>{s.name}</h3><small>{displayCategory(s.category)}</small><p>{s.description}</p></div><button className="button secondary" onClick={()=>importSample(s.id)} disabled={busy}>Import</button></div>):<div className="sample-empty">{busy?'Loading samples…':'No samples available from this server.'}</div>}</div></div></div>}
 </div>;
}
