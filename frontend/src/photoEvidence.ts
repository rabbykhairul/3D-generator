import type {Project,Photo} from './types';
export function armPhotos(project:Project):Photo[]{
 return ['left','right'].flatMap(side=>{
  const photo=project.photos.filter(p=>p.role==='fitting'&&[side,`inner-${side}`].includes(p.view)).at(-1);
  return photo?[photo]:[];
 });
}
