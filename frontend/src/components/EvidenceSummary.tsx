import type {MultiviewReport,Project,AnatomyReport} from '../types';
import AnatomySummary from './AnatomySummary';
import './evidenceSummary.css';

const STATUS_LABEL={used:'Used in model',accepted:'Aligned, not applied',review:'Needs review',skipped:'Not used'} as const;

function label(view:string){return view.replaceAll('-',' ').replace(/^\w/,letter=>letter.toUpperCase());}
function reasonText(reason:string){
 return ({
  insufficient_consensus:'Too few consistent feature matches to align this photo safely. Its pixels were not applied to the model.',
  too_few_matches:'Too few matching details to align this photo safely. Its pixels were not applied to the model.',
  insufficient_matches:'Too few matching details to align this photo safely. Its pixels were not applied to the model.',
 } as Record<string,string>)[reason]||reason;
}

export default function EvidenceSummary({project}:{project:Project}){
 const raw=project.artifact?.metadata?.multiview;
 const report:MultiviewReport|undefined=raw&&typeof raw==='object'&&Array.isArray(raw.views)?raw:undefined;
 if(!report)return <section className="review-card evidence-summary"><h3>Photo use</h3><p>This model has no per-photo use report. See the downloaded metadata for its available provenance; the presence of an uploaded photo alone does not mean it affected the model.</p></section>;
 const rows=report.views.filter(row=>row&&typeof row.photo_id==='string'&&typeof row.view==='string');
 const used=rows.filter(row=>row.status==='used').length;
 const matched=new Set(rows.map(row=>row.photo_id));
 const unreported=project.photos.filter(photo=>!matched.has(photo.id));
 return <section className="review-card evidence-summary" aria-label="Photo use in generated model">
  <h3>How this model used your photos</h3>
  <p>{used} of {rows.length} reported photo{rows.length===1?'':'s'} applied to the model. An aligned photo can still be excluded when its evidence is uncertain or no supported surface can use it.</p>
  <div className="evidence-rows">{rows.map((row,index)=>{
   const photo=project.photos.find(item=>item.id===row.photo_id);
   const status=row.status in STATUS_LABEL?row.status:'review';
   return <div className="evidence-row" key={`${row.photo_id}-${index}`}>
    {photo&&<img src={photo.url} alt=""/>}
    <div className="evidence-row-copy"><div className="evidence-row-heading"><strong>{label(row.view)}</strong><span className={`evidence-status ${status}`}>{status==='accepted'&&row.reason==='appearance_consistent'?'Checked, not applied':STATUS_LABEL[status]}</span></div>
     {row.usage&&<p>{row.usage}</p>}{row.reason&&<p>{reasonText(row.reason)}</p>}
     {photo&&<small>{photo.width} × {photo.height} px · {photo.role==='validation'?'held out':'fitting'} photo</small>}
    </div>
   </div>;
  })}</div>
  {unreported.length>0&&<p className="evidence-unreported">{unreported.length} uploaded photo{unreported.length===1?' has':'s have'} no entry in this report. Their contribution is unknown.</p>}
  {Array.isArray(report.limitations)&&report.limitations.length>0&&<div className="evidence-limitations"><strong>Remaining limits</strong><ul>{report.limitations.filter(item=>typeof item==='string').map((item,index)=><li key={index}>{item}</li>)}</ul></div>}
  <AnatomySummary report={project.artifact?.metadata?.anatomy as AnatomyReport|undefined}/>
 </section>;
}
