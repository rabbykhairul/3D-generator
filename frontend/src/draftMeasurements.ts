import type {Annotation,Measurement} from './types';

export const SIZE_KEYS=['frame_width','lens_width','lens_height','bridge_width','temple_length'] as const;
export function measurementIssue(values:Record<string,Measurement>):string|null{
 const depth=values.frame_thickness?.value;
 if(depth!==undefined&&(!Number.isFinite(depth)||depth<.8||depth>16))return 'Frame thickness is the front-to-back material depth, not the rim width in the front photo. Enter 0.8–16 mm, or leave it blank for an explicitly unverified draft.';
 for(const key of SIZE_KEYS){
  const v=values[key]?.value;if(v===undefined)continue;
  const minimum=key==='frame_width'||key==='temple_length'?10:key==='bridge_width'?.1:1;
  if(!Number.isFinite(v)||v<minimum||v>500)return `${key.replaceAll('_',' ')} is ${v} mm. Enter millimeters, not meters (0.2 m = 200 mm), or use approximate draft sizes if the real size is unknown.`;
 }
 const f=values.frame_width?.value,l=values.lens_width?.value,b=values.bridge_width?.value;
 if(f&&l&&b&&2*l+b>f*1.02)return `Two lens widths plus bridge (${(2*l+b).toFixed(2)} mm) exceed frame width (${f} mm). Correct the sizes or replace them with consistent approximate draft sizes.`;
 return null;
}

// No absolute physical size can be recovered from uncalibrated photos alone.
// Keep supplied values; mark every added number as default or inferred.
export function approximateMeasurements(annotations:Annotation[],existing:Record<string,Measurement>,resetSizes=false):Record<string,Measurement>{
 const result=structuredClone(existing);
 if(resetSizes)for(const key of SIZE_KEYS)delete result[key];
 const bounds=(part:string)=>{
  const points=annotations.find(a=>a.part===part)?.points;
  if(!points?.length)return null;
  const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
  return {minX:Math.min(...xs),maxX:Math.max(...xs),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys)};
 };
 const add=(key:string,value:number,source:Measurement['source'])=>{if(!(result[key]?.value>0)||(['inferred','default'].includes(result[key].source)&&!['frame_width','temple_length'].includes(key)))result[key]={value:Math.round(value*100)/100,source};};
 add('frame_width',140,'default');add('temple_length',140,'default');
 const sizeIssue=measurementIssue(Object.fromEntries(['frame_width','temple_length'].map(k=>[k,result[k]])));
 if(sizeIssue)throw new Error(sizeIssue);
 const left=bounds('lens-left'),right=bounds('lens-right'),front=bounds('front');
 if(left&&right){
  const width=front?.width??Math.max(left.maxX,right.maxX)-Math.min(left.minX,right.minX);
  if(width<=0)throw new Error('Please correct the front outline before estimating proportions.');
  const scale=result.frame_width.value/width;
  const ordered=[left,right].sort((a,b)=>a.minX-b.minX);
  add('lens_width',(left.width+right.width)/2*scale,'inferred');
  add('lens_height',(left.height+right.height)/2*scale,'inferred');
  const bridge=(ordered[1].minX-ordered[0].maxX)*scale;
  if(bridge<=0)throw new Error('Lens boundaries overlap horizontally. Correct the outlines before estimating bridge width.');
  add('bridge_width',bridge,'inferred');
 }else{
  add('lens_width',result.frame_width.value*.36,'default');
  add('lens_height',result.frame_width.value*.28,'default');
  add('bridge_width',result.frame_width.value*.13,'default');
 }
 return result;
}
