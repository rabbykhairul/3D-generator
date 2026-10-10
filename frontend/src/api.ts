import type {Project,Job,Annotation,Category,AnatomyReport} from './types';
export type ConstructionSuggestion={category:Category|null;confidence:'high'|'medium'|'low';reason:string;alternatives?:string[]};
export type DetectionReport={source_zoom?:{regions:{region:string;edge_support_fraction_within_2px:number;status:string}[];outline_changed:boolean};id:string;view?:string;status:'reference-only'|'needs-correction'|'partial'|'review-required';missing_parts:string[];component_evidence?:{accepted_parts:string[];registered_parts:string[];extra_views:{view:string;usage?:string;reason?:string}[]};recognizer:{backend:string;learned_model_active:boolean;proposals_used?:boolean;experimental?:boolean;status?:string;model?:string;prompt_source?:string;reason:string};next_action:string;fidelity_verified:boolean;automatic_acceptance:boolean};
async function request<T>(path:string, options:RequestInit={}):Promise<T>{
 const response=await fetch('/api'+path,options);
 if(!response.ok){const e=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail));}
 return response.json();
}
export const api={
 health:()=>request<{engine_version?:number;joint_front_fit_available?:boolean}>('/health'),
 list:()=>request<Project[]>('/projects'),
 archived:()=>request<Project[]>('/projects?archived=true'),
 removeProject:(id:string)=>request<Project>(`/projects/${id}`,{method:'DELETE'}),
 restoreProject:(id:string)=>request<Project>(`/projects/${id}/restore`,{method:'POST'}),
 create:(name:string,category:string)=>request<Project>('/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,category})}),
 get:(id:string)=>request<Project>(`/projects/${id}`),
 patch:(id:string,data:Partial<Project>)=>request<Project>(`/projects/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}),
 upload:(id:string,file:File,view:string,role='fitting')=>{const body=new FormData();body.append('file',file);body.append('view',view);body.append('role',role);return request<Project>(`/projects/${id}/photos`,{method:'POST',body});},
 removePhoto:(id:string,photoId:string)=>request<Project>(`/projects/${id}/photos/${photoId}`,{method:'DELETE'}),
 relabelPhoto:(id:string,photoId:string,view:string)=>request<Project>(`/projects/${id}/photos/${photoId}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({view})}),
 extract:(id:string,photoId:string,backend?:'heuristics'|'cpu-metal'|'hybrid'|'sam2'|'sam3')=>request<{annotations:Annotation[];warnings:string[];anatomy?:AnatomyReport;detection_report?:DetectionReport;construction:ConstructionSuggestion;appearance_evidence?:{part:string;points:{x:number;y:number}[];possible_mark_regions:number;reason:string;applied:boolean}[]}>(`/projects/${id}/photos/${photoId}/extract${backend?`?backend=${backend}`:''}`,{method:'POST'}),
 detectionRuns:(id:string)=>request<DetectionReport[]>(`/projects/${id}/detection-runs`),
 auditAnatomy:(id:string,photoId:string,annotations:Annotation[],category:Category)=>request<AnatomyReport>(`/projects/${id}/photos/${photoId}/anatomy`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({annotations,category})}),
 generate:(id:string)=>request<Job>(`/projects/${id}/jobs`,{method:'POST'}),
 job:(id:string)=>request<Job>(`/jobs/${id}`),
 samples:()=>request<{id:string;name:string;category:string;kind:string;description:string}[]>('/samples'),
 importSample:(id:string)=>request<Project>(`/samples/${id}/import`,{method:'POST'})
};
