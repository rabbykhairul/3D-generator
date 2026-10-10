import './photoGuide.css';

const EXAMPLES = [
 {view:'front',image:'front',label:'1 · Front',badge:'Start here',alt:'Straight front view of a pale transparent frame, with both complete lens rims visible',
  description:'Both rims, bridge and outside corners fully visible. Keep the camera level with the glasses.',action:'Upload front photo'},
 {view:'right',image:'side',label:'2 · Either side',badge:'Build an arm',alt:'Side-on glasses showing the full near arm from its hinge to curved ear tip',
  description:'Show the hinge and entire arm through the ear tip. Use Left or Right to match your photo.',action:'Upload side photo'},
 {view:'front-left',image:'angle',label:'3 · Three-quarter',badge:'Optional reference',alt:'Open glasses seen at a three-quarter angle showing the front and both arms',
  description:'Helps check alignment and appearance across views when it can be matched to the front. It does not replace a straight front or side outline.',action:'Upload angled reference'},
 {view:'inner',image:'',label:'4 · Inner / rear',badge:'Optional back surface',alt:'',
  description:'Can inform the visible inner or back surface if the view registers with approved geometry. Otherwise it remains a comparison reference.',action:'Upload inner photo'},
] as const;

export default function PhotoGuide({onChoose,disabled=false}:{onChoose:(view:string)=>void;disabled?:boolean}){
 return <details className="photo-guide" open>
  <summary>What should I upload? <span>See real photo examples</span></summary>
  <div className="photo-guide-content">
   <p className="photo-guide-intro">Use existing supplier photos of the <strong>same product and color</strong>. Start with the front; add a side photo for an arm. You do not need a twelve-view photo shoot.</p>
   <div className="photo-guide-examples">{EXAMPLES.map(example=><article key={example.view}>
    <div className="photo-guide-example-heading"><h4>{example.label}</h4><span>{example.badge}</span></div>
    {example.image?<img src={`/upload-guide/${example.image}.png`} width="1080" height="1080" alt={example.alt} loading="lazy"/>:<div className="photo-guide-no-example">Photograph the open frame from the inside, with both inner rims and the bridge visible.</div>}
    <p>{example.description}</p>
    <button type="button" className="button secondary" disabled={disabled} onClick={()=>onChoose(example.view)}>{example.action}</button>
   </article>)}</div>
   <p className="photo-guide-note">One side can supply the opposite arm only if you explicitly approve mirroring. Upload both sides when their construction differs. A front-only draft has no verified arms. For clear plastic like these examples, choose Transparent full rim in Frame construction; this controls the material approximation, not automatic approval.</p>
   <div className="photo-guide-tips">
    <section><h4>Best chance of a quick result</h4><ul>
     <li>Whole product in the image, in focus, without heavy compression.</li>
     <li>Plain, contrasting background and soft, even lighting.</li>
     <li>Prefer 1000 px or more on the longest edge; use the original image, not a shop thumbnail.</li>
     <li>For clear or pale frames, a matte light-gray background helps reveal the edges. Do not paint over the frame or lens openings.</li>
    </ul></section>
    <section><h4>What may still need review</h4><ul>
     <li>Clear rims, reflections and dark lenses can make boundaries uncertain.</li>
     <li>Hands, faces, packaging or a busy background can hide real frame edges.</li>
     <li>Cropped close-ups are useful references, but cannot provide missing geometry.</li>
     <li>Different angles are welcome in the correct slots; an angled image cannot substitute for a straight front in this demo.</li>
    </ul></section>
   </div>
   <p className="photo-guide-note">JPEG / PNG / WebP · up to 20 MB. Small images are accepted with quality warnings and may be enlarged for analysis, but enlargement does not restore detail. Higher-resolution originals remain available for inspection. Automatic suggestions always need approval; photos alone do not establish physical size, hidden geometry or exact material transparency.</p>
  </div>
 </details>;
}
