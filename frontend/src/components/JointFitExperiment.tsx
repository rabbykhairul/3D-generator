import {useState} from 'react';
import {api} from '../api';
import type {Project} from '../types';

type Props={project:Project;onUpdate:(project:Project)=>void;onError:(message:string)=>void};
export default function JointFitExperiment({project,onUpdate,onError}:Props){
 const [busy,setBusy]=useState(false);
 const enabled=project.reconstruction_settings?.joint_front_fit??false;
 const fit=project.artifact?.metadata?.fit?.joint_front;
 const review=project.artifact?.metadata?.export_review;
 async function toggle(){setBusy(true);onError('');try{const health=await api.health();if(!health.joint_front_fit_available||!health.engine_version||health.engine_version<44)throw new Error('Restart the API and worker to load engine 44 before testing multi-view fitting.');onUpdate(await api.patch(project.id,{reconstruction_settings:{joint_front_fit:!enabled}}));}catch(e){onError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 const url=(file:string)=>project.artifact?.metadata_url.replace(/metadata\.json$/,file)??'';
 return <section className="review-card" aria-label="Experimental multi-view reconstruction">
  <h3>Multi-view front fitting · experiment</h3>
  <p>Uses front-left / front-right fitting photos to jointly fit front wrap and camera angles. Arms, thickness and hidden hardware still use existing geometry. Approved outlines remain required.</p>
  <label><input type="checkbox" checked={enabled} disabled={busy} onChange={()=>void toggle()}/> Enable experiment for the next generation</label>
  <p>Upload at least one oblique view (both are better), enable this option, then Generate / Regenerate. Clear product details must match between photos. Existing measured front curvature or approved top-view geometry takes precedence.</p>
  {fit&&!project.stale&&<><h4>Fit result: {fit.status}</h4><p>{fit.reason}</p>{typeof fit.fitted_front_curve_mm==='number'&&<p>Front wrap: {fit.initial_front_curve_mm} → {fit.fitted_front_curve_mm} mm. Withheld feature error: {fit.baseline_withheld_median_error_px} → {fit.withheld_median_error_px} px. This is a fit diagnostic, not a product accuracy score.</p>}
   {Array.isArray(fit.views)&&fit.views.length>0&&<ul>{fit.views.map((v:any,i:number)=><li key={i}>{v.view}: {v.matches} matches · {v.withheld_matches} withheld · {v.withheld_median_error_px} px residual</li>)}</ul>}
  </>}
  {review&&!project.stale&&<><h4>Serialized GLB checks</h4><p>Approved components preserved: {review.component_preservation_passed?'yes':'no'}. Geometry count: {review.serialized_geometry_count}.{typeof review.front_trace_iou==='number'&&` Front trace overlap: ${(review.front_trace_iou*100).toFixed(1)}% (construction consistency only).`}</p>
   <div className="reference-strip">{review.images?.map((item:{file:string;label:string})=><figure key={item.file} style={{margin:0,maxWidth:420}}><a href={url(item.file)} target="_blank" rel="noreferrer"><img src={url(item.file)} alt={item.label} style={{width:'100%',height:'auto',maxHeight:260,objectFit:'contain'}}/></a><figcaption style={{fontSize:12}}>{item.label}</figcaption></figure>)}</div>
  </>}
  <p>Insufficient evidence keeps the current shape. Zoomed source crops show outline support; they do not recover unseen surfaces or certify optical fit.</p>
 </section>;
}
