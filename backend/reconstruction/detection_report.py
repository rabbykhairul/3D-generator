"""Operational detection reports, not recognition accuracy certificates."""
from .anatomy import FRONT_VIEWS, SIDE_VIEWS, audit_photo
from ..engine_version import ENGINE_VERSION


def detection_report(photo, result):
    annotations = result.get('annotations', [])
    audit = result.get('anatomy') or audit_photo(photo, annotations)
    found = {a.get('part') for a in annotations}
    view = photo.get('view', '').lower()
    category = (result.get('construction') or {}).get('category')
    required = []
    if view in FRONT_VIEWS:
        required = ['shield'] if category == 'sports-shield' else ['lens-left', 'lens-right']
        if category and category not in {'rimless', 'sports-shield'}:
            required.append('front')
    elif view in SIDE_VIEWS:
        required = ['temple-left' if 'left' in view else 'temple-right']
    missing = [p for p in required if p not in found]
    reference_only = view not in FRONT_VIEWS | SIDE_VIEWS
    status = ('reference-only' if reference_only else
              'needs-correction' if not annotations or audit.get('blocking') else
              'partial' if missing or result.get('hybrid_detection',{}).get('status')=='partial' else 'review-required')
    recognizer = {'backend': 'geometric-heuristics', 'learned_model_active': False,
                  'reason': 'Contour detection is active; pretrained lens proposals are available through the experimental hybrid option.'}
    cpu = result.get('cpu_metal_detection')
    rimless_gradient = result.get('segmentation',{}).get('method') == 'cpu-gradient-rimless-source-edges'
    if cpu or rimless_gradient:
        recognizer = {'backend': 'cpu-thin-metal' if cpu and cpu['applied'] else 'cpu-gradient-rimless' if rimless_gradient else 'geometric-heuristics',
                      'learned_model_active': False, 'experimental': False,
                      'reason': 'Native-source CPU thin-metal boundaries and connected support used; no AI inference ran.' if cpu and cpu['applied'] else
                                'Stable lens edges and bridge mounts supported a CPU rimless draft; no AI inference ran.' if rimless_gradient else
                                'CPU metal checks declined this photo; standard CPU contours retained. No AI inference ran.'}
    hybrid = result.get('hybrid_detection')
    if hybrid:
        recognizer = dict(backend=(hybrid['requested_backend']+'-source-proposals' if hybrid.get('requested_backend') in {'sam2','sam3','eyewear-trained'} else 'hybrid-lens-proposals') if hybrid.get('proposals_used') else 'geometric-heuristics',
                          model=hybrid.get('proposal_backend'),
                          prompt_source=hybrid.get('runtime',{}).get('prompt_source'),
                          learned_model_active=bool(hybrid.get('inference_ran')),
                          proposals_used=bool(hybrid.get('proposals_used')),experimental=True,
                          status=hybrid['status'],reason=hybrid['reason'])
    components=result.get('component_detection',{})
    component_evidence=dict(accepted_parts=[a['part'] for a in components.get('accepted',[])],
                            rejected_regions=components.get('rejected',[]),
                            registered_parts=result.get('component_multiview',{}).get('accepted_parts',[]),
                            extra_views=result.get('component_multiview',{}).get('views',[]),
                            depth_measured=False)
    return dict(schema_version=1, engine_version=ENGINE_VERSION, photo_id=photo['id'],
                view=view, status=status, suggested_parts=sorted(found), missing_parts=missing,
                issues=audit.get('issues', []),
                sources=sorted({a.get('source', 'unspecified') for a in annotations}),
                recognizer=recognizer, component_evidence=component_evidence,
                next_action=('Keep as reference; this view does not supply automatic fitting outlines.'
                             if reference_only else
                             'Correct the missing/conflicting boundaries in Advanced; the saved upload can be replayed locally.'
                             if status in {'partial', 'needs-correction'} else
                             'Review outlines, construction and attachment anchors before approval.'),
                warnings=result.get('warnings', []), source_zoom=result.get('source_zoom'),
                fidelity_verified=False, automatic_acceptance=False)
