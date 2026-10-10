import {useEffect,useState} from 'react';
import {api} from '../api';
import type {Material,Measurement,Project} from '../types';
import {approximateMeasurements,measurementIssue} from '../draftMeasurements';

const FIELDS:[string,string,string,boolean][]=[
 ['frame_width','Overall frame width','Outer edge to outer edge',true],
 ['lens_width','Lens width','Widest horizontal span of one lens',true],
 ['lens_height','Lens height','Tallest vertical span of one lens',true],
 ['bridge_width','Bridge width','Shortest distance between lenses',true],
 ['temple_length','Temple length','Hinge to temple tip',true],
 ['frame_thickness','Frame thickness','Front-to-back material depth, not visible rim width (0.8–16 mm)',false],
 ['temple_thickness','Temple thickness','Typical arm thickness',false],
 ['front_curve','Front curve','Front wrap depth, if measured',false],
 ['temple_spread','Temple spread','Inward offset of each arm from its hinge',false]
];
const SOURCES:Measurement['source'][]=['measured','published','manual','inferred','default'];
type Props={project:Project;onUpdate:(project:Project)=>void;onError:(message:string)=>void};

export default function Measurements({project,onUpdate,onError}:Props){
 const [values,setValues]=useState<Record<string,Measurement>>(structuredClone(project.measurements));
 const [material,setMaterial]=useState<Material>({...project.material});
 const [notes,setNotes]=useState(project.notes||'');
 const [busy,setBusy]=useState(false);
 const [dirty,setDirty]=useState(false);
 useEffect(()=>{setValues(structuredClone(project.measurements));setMaterial({...project.material});setNotes(project.notes||'');setDirty(false);},[project.id,project.measurements,project.material,project.notes]);
 function setValue(key:string,value:string){setDirty(true);setValues(v=>({...v,[key]:{value:Number(value),source:'manual'}}));}
 const front=project.photos.filter(p=>p.view==='front'&&p.role==='fitting').at(-1);
 const frontAnnotations=project.annotations.filter(a=>a.photo_id===front?.id&&a.approved);
 function resetSizes(){try{setValues(approximateMeasurements(frontAnnotations,values,true));setDirty(true);onError('');}catch(e){onError(e instanceof Error?e.message:String(e));}}
 function setSource(key:string,source:Measurement['source']){setDirty(true);setValues(v=>({...v,[key]:{value:v[key]?.value||0,source}}));}
 function setMat<K extends keyof Material>(key:K,value:Material[K]){setDirty(true);setMaterial(m=>({...m,[key]:value}));}
 async function save(){
  const zeroAllowed=(key:string)=>['front_curve','temple_spread'].includes(key);
  const measurements=Object.fromEntries(Object.entries(values).filter(([key,v])=>Number.isFinite(v.value)&&(v.value>0||(zeroAllowed(key)&&v.value===0))));
  const bad=Object.entries(values).find(([key,v])=>!Number.isFinite(v.value)||(zeroAllowed(key)?v.value<0:v.value<=0)||v.value>500);
  if(bad){onError(`${bad[0].replaceAll('_',' ')} must be between 0 and 500 mm, or left blank.`);return;}
  if(measurements.frame_width?.value!==project.measurements.frame_width?.value&&measurements.frame_width?.value>=10&&frontAnnotations.length){
   try{const refreshed=approximateMeasurements(frontAnnotations,measurements);for(const key of Object.keys(measurements))if(measurements[key].source==='inferred')measurements[key]=refreshed[key];}catch(e){onError(e instanceof Error?e.message:String(e));return;}
  }
  const issue=measurementIssue(measurements);if(issue){onError(issue);return;}
  if(measurements.bridge_width&&measurements.frame_width&&measurements.bridge_width.value>=measurements.frame_width.value){onError('Bridge width must be smaller than overall frame width.');return;}
  setBusy(true);try{onUpdate(await api.patch(project.id,{measurements,material,notes}));setDirty(false);}catch(e){onError(String(e instanceof Error?e.message:e));}finally{setBusy(false);}
 }
 const complete=FIELDS.filter(([key,,,required])=>required&&values[key]?.value>0).length;
 const depth=values.frame_thickness;
 return <section className="step-content dimensions-page"><div className="section-heading"><div><span className="eyebrow">03 / Scale & appearance</span><h2>Dimensions & materials</h2><p>Enter millimeters from a ruler, product specifications, or a clearly labeled estimate. Source labels travel with the model.</p></div><div className="section-count"><strong>{complete}<span> / 5</span></strong><small>required dimensions</small></div></div>
  <div className="notice warning"><p>All fields use <strong>millimeters</strong>. For example, 0.2 meters means 200 mm—not 0.2 mm. If size is unknown, the button below replaces the five main size values with a 140 mm draft scale and photo-derived proportions where available. These are estimates, not detected physical measurements.</p>{measurementIssue(values)&&<p role="alert">{measurementIssue(values)}</p>}<button className="button secondary" onClick={resetSizes} disabled={busy}>Use approximate draft sizes (140 mm)</button><p>Review the new values, then Save dimensions & appearance. Your photos, outlines and materials are retained.</p></div>
  <div className="notice"><h3>Frame depth controls the side profile</h3><p>{depth?<>Requested front depth: <strong>{depth.value} mm</strong> ({depth.source}).</>:<>No frame depth supplied: solid-frame drafts use <strong>4 mm, unverified</strong>.</>} The front photo establishes the rim outline, not its front-to-back thickness. Thin-metal construction additionally limits wire diameter to the available rim clearance.</p><p>To correct a bulky frame, measure its material depth from the side or enter a clearly labeled estimate in Frame thickness below. Save, then regenerate; changing AR scale cannot correct material thickness. Estimates do not establish physical fit.</p></div>
  <div className="dimension-layout"><div className="dimensions-card"><div className="card-title"><h3>Physical dimensions</h3><span>mm</span></div><div className="measurement-head"><span>Measurement</span><span>Value</span><span>Provenance</span></div>{FIELDS.map(([key,label,hint,required])=><div className="measurement-row" key={key}><div><label htmlFor={`measure-${key}`}>{label}{required&&<em>required</em>}</label><small>{hint}</small></div><div className="number-unit"><input id={`measure-${key}`} type="number" min="0" max="500" step="0.1" value={values[key]?.value??''} onChange={e=>{if(!e.target.value){setDirty(true);setValues(v=>{const next={...v};delete next[key];return next;});}else setValue(key,e.target.value);}} placeholder="—"/><span>mm</span></div><select value={values[key]?.source||'manual'} onChange={e=>setSource(key,e.target.value as Measurement['source'])} aria-label={`${label} source`}>{SOURCES.map(source=><option value={source} key={source}>{source}</option>)}</select></div>)}<div className="dimension-foot"><strong>Provenance matters</strong><p>“Measured” means you checked the physical object. “Published” means a seller or maker supplied the value. Use inferred or default only when the evidence is missing.</p></div></div>
   <div className="materials-card"><div className="card-title"><h3>Appearance</h3><span>Approximation</span></div><div className="material-preview" style={{background:`linear-gradient(135deg, ${material.color}, ${material.lens_color})`}}><div className="preview-lens" style={{background:material.lens_color,opacity:material.lens_opacity}}/></div>
    <div className="color-field"><label htmlFor="frame-color">Frame color</label><input id="frame-color" type="color" value={material.color} onChange={e=>setMat('color',e.target.value)}/><code>{material.color}</code></div><div className="color-field"><label htmlFor="lens-color">Lens tint</label><input id="lens-color" type="color" value={material.lens_color} onChange={e=>setMat('lens_color',e.target.value)}/><code>{material.lens_color}</code></div>
    {([['roughness','Roughness'],['metalness','Metalness'],['opacity','Frame opacity'],['lens_opacity','Lens opacity']] as const).map(([key,label])=><label className="range-field" key={key}><span>{label}<strong>{Math.round(material[key]*100)}%</strong></span><input type="range" min="0" max="1" step="0.01" value={material[key]} onChange={e=>setMat(key,Number(e.target.value))}/></label>)}
    <label className="field notes-field">Construction notes<textarea rows={5} value={notes} onChange={e=>{setDirty(true);setNotes(e.target.value);}} placeholder="E.g. acetate front, metal hinges, visible nose pads…"/></label><p className="material-note">Color and transparency are estimated from photos. They are not a measured optical material specification.</p>
   </div></div><div className="sticky-actions"><span>{dirty?'Unsaved changes':'All changes saved'}</span><button className="button primary" onClick={save} disabled={busy||!dirty}>{busy?'Saving…':'Save dimensions & appearance'}</button></div>
 </section>;
}
