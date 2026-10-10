import type {DetectionReport} from '../api';

export default function DetectionStatus({report}:{report:DetectionReport|null}){
 if(!report)return null;
 return <div className="notice warning" role="status">
  <strong>Detection: {report.status.replaceAll('-',' ')}</strong>
  {report.recognizer.model&&<p>Segmentation model: {report.recognizer.model}{report.recognizer.prompt_source==='automatic-eyewear-text-concepts'?' · automatic eyewear text prompts':report.recognizer.prompt_source==='coarse-lens-locator'?' · automatic lens prompts':report.recognizer.prompt_source==='automatic-photo-grid'?' · automatic image grid':report.recognizer.prompt_source==='automatic-photo-grid-and-branch-prompts'?' · automatic image grid and arm branch prompts':''}</p>}
  {report.recognizer.experimental&&report.recognizer.backend!=='cpu-thin-metal'&&<p>{report.recognizer.proposals_used?`AI assisted ${['left','right','left-side','right-side','side-left','side-right','inner-left','inner-right'].includes(report.view??'')?'arm':'lens'} boundaries used · experimental`:report.recognizer.learned_model_active?report.recognizer.status==='rejected'?'AI proposals did not pass boundary checks · experimental':'AI proposals checked; standard geometry retained · experimental':'AI assisted analysis was not applied; standard analysis retained.'}</p>}
  <p>{report.recognizer.reason}</p>
  {!!report.component_evidence?.accepted_parts.length&&<p>Added source-checked components: {report.component_evidence.accepted_parts.map(p=>p.replaceAll('-',' ')).join(', ')}. Component depth remains inferred.</p>}
  {!!report.component_evidence?.registered_parts.length&&<p>Extra-view proposals: {report.component_evidence.registered_parts.map(p=>p.replaceAll('-',' ')).join(', ')}. Their mapped position and depth are inferred.</p>}
  {report.missing_parts.length>0&&<p>Missing fitting boundaries: {report.missing_parts.join(', ')}. Missing detection does not mean the product has no such part.</p>}
  {!!report.source_zoom?.regions.length&&<details><summary>Native-source zoom checks · frame / hinge / tip</summary><p>Local source crops were checked at native resolution. Edge support can include shadows or reflections and is not a recognition score.</p><ul>{report.source_zoom.regions.map((region,index)=><li key={index}>{region.region}: {(region.edge_support_fraction_within_2px*100).toFixed(0)}% nearby edge support · {region.status}</li>)}</ul><p>Use Advanced zoom + to inspect ownership before approval. Full close-up overlays are included after generation with the multi-view experiment enabled.</p></details>}
  <p>{report.next_action}</p>
  <small>Local diagnostic saved. This is not a fidelity pass or automatic approval.</small>
 </div>;
}
