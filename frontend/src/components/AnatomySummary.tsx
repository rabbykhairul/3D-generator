import type {AnatomyReport} from '../types';

export default function AnatomySummary({report}:{report:AnatomyReport|null|undefined}){
 if(!report)return null;
 const parts=report.nodes.filter(node=>node.state!=='not-observed-in-view'&&node.state!=='not-applicable');
 return <section className="review-card" aria-label="Eyewear structure evidence">
  <h4>Eyewear structure · {report.status.replaceAll('-',' ')}</h4>
  <p>{report.scope==='photo'?`Evidence from this ${report.view??''} photo only. Components verified in other photos appear in their own reports.`:'Evidence across the project’s reviewed photos.'}</p>
  <p>These are geometry checks, not a universal AI recognition score. Observed means a traced boundary or anchor; inferred means a shape or symmetry assumption. Neither means the part is approved or its hidden depth is measured.</p>
  <dl style={{display:'grid',gridTemplateColumns:'minmax(140px, 1fr) 1fr',gap:'6px 16px'}}>{parts.map(node=><div key={node.id} style={{display:'contents'}}><dt>{node.id.replaceAll('-',' ')}</dt><dd style={{margin:0}}>{node.state}{node.evidence.length>0?node.evidence.every(e=>e.approved)?' · reviewed':' · review required':''}</dd></div>)}</dl>
  {report.issues.length>0&&<details open={report.blocking}><summary>{report.issues.length} structure finding{report.issues.length===1?'':'s'}{report.blocking?' · correct geometry before approval':''}</summary><ul>{report.issues.map((issue,index)=><li key={index}>{issue.message} {issue.action}</li>)}</ul></details>}
  {report.guidance?.map((item,index)=><p key={index}>{item}</p>)}
 </section>;
}
