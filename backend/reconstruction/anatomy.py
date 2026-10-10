"""CPU structural hypotheses, not semantic recognition or confidence probabilities.

The graph is independent of SKU, finish and contour style. Geometry checks are
necessary but not sufficient evidence that an arbitrary photograph is eyewear.
Missing detection never establishes absence, occlusion or a hardware design.
"""
from __future__ import annotations

import math
from shapely.geometry import Point, Polygon

SCHEMA_VERSION = 1
FRONT_VIEWS = {'front', 'frontal', 'straight-front', 'front-on'}
SIDE_VIEWS = {'left', 'right', 'left-side', 'right-side', 'side-left',
              'side-right', 'inner-left', 'inner-right'}
PARTS = ('front', 'front-support', 'lens-left', 'lens-right', 'shield', 'bridge',
         'bridge-opening', 'bridge-hardware', 'hinge-left', 'hinge-right',
         'lens-mount-left', 'lens-mount-right',
         'temple-left', 'temple-right', 'nose-pad-left', 'nose-pad-right')


def structure_schema():
    """Public ontology; independent of specific shape, colour or merchant SKU."""
    return dict(schema_version=SCHEMA_VERSION, parts=list(PARTS),
                construction_families=['opaque-full-rim', 'thin-metal', 'transparent-full-rim',
                                      'semi-rimless', 'rimless', 'sports-shield'],
                states=['observed', 'inferred', 'uncertain', 'not-applicable', 'not-observed-in-view'],
                optical_topologies=['separate-left-right', 'single-shield'],
                arm_structure=['hinge-attachment', 'shaft', 'bend', 'tip', 'inner-surface', 'outer-surface'],
                coordinate_system='original photo pixels; lens-left/right name source-image halves',
                rules=['Use freeform optical contours, not a catalogue of SKU-specific shapes.',
                       'Separate lens, frame support, bridge, nose support and arm evidence.',
                       'Single-arm geometry needs a coherent hinge-to-tip span.',
                       'Shaft/bend depth and hidden surfaces are not measured by a 2D contour.',
                       'Missing detection is not proof of absence, rimless construction or occlusion.',
                       'Symmetry and occlusion completion remain labelled assumptions.'],
                recognizer='geometric audits; optional pretrained lens proposals are reported per extraction')


def _polygon(annotation, width, height):
    points = annotation.get('points', [])
    if len(points) < 3:
        return None, 'A closed outline needs at least three points.'
    try:
        coords = [(float(p['x']), float(p['y'])) for p in points]
        if any(not math.isfinite(x) or not math.isfinite(y) or
               x < 0 or y < 0 or x > width or y > height for x, y in coords):
            return None, 'Outline coordinates are outside the source image or non-finite.'
        polygon = Polygon(coords)
        if not polygon.is_valid or polygon.area <= 0:
            return None, 'Outline self-intersects or has no enclosed area.'
        return polygon, None
    except (KeyError, TypeError, ValueError):
        return None, 'Outline coordinates are malformed.'


def audit_photo(photo, annotations, category=None, *, width=None, height=None):
    """Recompute from current geometry; never trust stale stored detection evidence."""
    width = width or photo.get('width', 0)
    height = height or photo.get('height', 0)
    view = str(photo.get('view', 'front')).lower()
    photo_id = str(photo.get('id', ''))
    selected = [a for a in annotations if str(a.get('photo_id')) == photo_id]
    issues, polygons, indexed = [], {}, {}

    def issue(code, part, message, action, blocking=False):
        issues.append(dict(code=code, part=part, message=message,
                           action=action, blocking=blocking))

    for annotation in selected:
        part = annotation.get('part', '')
        if part in indexed:
            issue('duplicate-part', part, 'Multiple outlines claim the same part in this photo.',
                  'Keep one active outline for this part.', True)
            continue
        indexed[part] = annotation
        polygon, error = _polygon(annotation, width, height)
        if error:
            issue('invalid-outline', part, error, 'Correct the outline in Advanced.', True)
        else:
            polygons[part] = polygon
        anchors = annotation.get('anchors', {})
        for name, point in anchors.items():
            try:
                x, y = float(point['x']), float(point['y'])
                valid = math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                issue('invalid-anchor', part, f'{name} anchor is outside the source image or malformed.',
                      'Place this anchor on visible hardware.', True)
        if part.startswith('temple-') and 'occlusion-completion' in annotation.get('source',''):
            issue('arm-overlap-inferred',part,
                  'The visible arm branch is separated, but ownership and thickness of its overlapping shaft are inferred.',
                  'Review the selected tip and shared shaft; hidden hinge geometry remains unmeasured.')
        if part.startswith('lens-') and 'occlusion-completion' in annotation.get('source',''):
            issue('lens-boundary-inferred',part,
                  'The optical outline includes a completed boundary across a photographed obstruction.',
                  'Review the completed sections and any unusual concave lens design.')
        if part.startswith('temple-'):
            missing = [name for name in ('hinge', 'tip') if name not in anchors]
            if missing:
                issue('missing-arm-anchor', part, f"Missing {', '.join(missing)} anchor.",
                      'Mark the visible hinge and the selected arm tip.', True)
            elif polygon is not None:
                valid_points = []
                for name in ('hinge', 'tip'):
                    try:
                        point = Point(float(anchors[name]['x']), float(anchors[name]['y']))
                        valid_points.append(point)
                        if polygon.distance(point) > max(width, height) * .01:
                            issue('arm-anchor-disconnected', part, f'{name} is disconnected from the arm outline.',
                                  'Correct the selected branch; do not join different arm tips.', True)
                    except (KeyError, TypeError, ValueError):
                        pass
                if len(valid_points) == 2 and valid_points[0].distance(valid_points[1]) < max(width, height) * .03:
                    issue('arm-anchor-span', part, 'Hinge and tip do not establish a usable arm span.',
                          'Place anchors at opposite ends of the same arm.', True)
        if view in SIDE_VIEWS and part in {'front', 'lens-left', 'lens-right', 'shield'}:
            issue('view-part-conflict', part, 'Side evidence cannot establish a frontal optical boundary.',
                  'Use a straight front photo for optical boundaries.', True)
        if view in FRONT_VIEWS and part.startswith('temple-'):
            issue('view-part-conflict', part, 'A folded arm in a front photo is not a side-on arm profile.',
                  'Use a separate side photo for arm geometry.', True)

    left, right = polygons.get('lens-left'), polygons.get('lens-right')
    front = polygons.get('front')
    if left is not None and right is not None:
        if left.intersection(right).area > 0:
            issue('overlapping-optical-boundaries', 'lenses', 'Left and right optical areas overlap.',
                  'Separate the optical boundaries; check view and branch selection.', True)
        # Part labels describe image halves, not measured 3D wearer coordinates.
        if left.centroid.x >= right.centroid.x:
            issue('reversed-optical-pair', 'lenses', 'Lens labels are reversed in source-image coordinates.',
                  'Assign lens-left to the left half of the source photo.', True)
        if front is not None:
            for part, polygon in [('lens-left', left), ('lens-right', right)]:
                # Subpixel contour antialiasing is not a structural defect.
                tolerance = max(width, height) * .0005
                if not front.buffer(tolerance).covers(polygon):
                    issue('optical-area-outside-support', part, 'The optical boundary extends beyond the outer frame.',
                          'Review inner and outer edges together; remove shadows.', True)
    for part in ('bridge-opening',):
        polygon = polygons.get(part)
        if polygon is not None and front is not None:
            if not front.buffer(max(width, height) * .0005).covers(polygon) or any(
                    polygon.intersection(p).area > 0 for p in (left, right) if p is not None):
                issue('invalid-bridge-opening', part, 'Bridge opening is outside support or intersects a lens.',
                      'Trace only the observed opening between bridge bars.', True)

    nodes = []
    for part in PARTS:
        annotation = indexed.get(part)
        relevant = (view in FRONT_VIEWS and not part.startswith('temple-') or
                    view in SIDE_VIEWS and (part.startswith('temple-') or part.startswith('hinge-')) and
                    part.endswith('left' if 'left' in view else 'right'))
        state = 'uncertain' if relevant else 'not-observed-in-view'
        evidence, source = [], None
        if annotation:
            source = annotation.get('source', 'manual')
            invalid = any(i['blocking'] and (i['part'] == part or
                          i['part'] == 'lenses' and part.startswith('lens-')) for i in issues)
            inferred = any(marker in source for marker in ('mirrored', 'occlusion-completion', 'view-registration'))
            state = 'uncertain' if invalid else 'inferred' if inferred else 'observed'
            evidence.append(dict(photo_id=photo_id, source=source,
                                 approved=bool(annotation.get('approved')),
                                 geometry_valid=part in polygons))
        if part == 'shield' and left is not None and right is not None:
            state = 'not-applicable'
        if part.startswith('lens-') and 'shield' in polygons:
            state = 'not-applicable'
        if part == 'front' and category == 'rimless' and annotation is None:
            state = 'not-applicable'
        if part == 'bridge' and left is not None and right is not None:
            # A connecting contour implies a structural hypothesis, not a
            # separately measured bridge shape or material.
            supporting = indexed.get('bridge-hardware') or indexed.get('front-support') or indexed.get('front')
            if supporting and supporting.get('part') in polygons:
                state = 'observed' if supporting['part']=='bridge-hardware' and 'view-registration' not in supporting.get('source','') else 'inferred'
                evidence = [dict(photo_id=photo_id, source=supporting.get('source', 'manual'),
                                 approved=bool(supporting.get('approved')), geometry_valid=True)]
        if part.startswith('hinge-'):
            side = part.removeprefix('hinge-')
            parent = indexed.get(f'temple-{side}') or indexed.get('front')
            name = 'hinge' if parent and parent['part'].startswith('temple-') else f'hinge-{side}'
            if parent and name in parent.get('anchors', {}):
                state = 'inferred' if ('mirrored' in parent.get('source', '') or
                    parent.get('anchor_provenance',{}).get(name,'').startswith('inferred')) else 'observed'
                if any(i['blocking'] and i['part'] == parent['part'] for i in issues):
                    state = 'uncertain'
                evidence = [dict(photo_id=photo_id, source='anchor:' + parent.get('source', 'manual'),
                                 approved=bool(parent.get('approved')), geometry_valid=parent['part'] in polygons)]
        nodes.append(dict(id=part, state=state, evidence=evidence))

    optical = left is not None and right is not None or 'shield' in polygons
    support = front is not None or 'bridge-hardware' in polygons or 'front-support' in polygons or 'shield' in polygons
    arm = any(p.startswith('temple-') for p in polygons)
    blocking = any(i['blocking'] for i in issues)
    recognized = view in FRONT_VIEWS and optical and support or view in SIDE_VIEWS and arm
    status = 'supported-structure' if recognized and not blocking else 'partial-structure' if polygons else 'unverified'
    if view not in FRONT_VIEWS | SIDE_VIEWS:
        status = 'reference-only'
    if view in FRONT_VIEWS and not optical:
        issue('optical-boundaries-unverified', 'lenses', 'A complete optical area was not established.',
              'Provide a sharp straight front photo or correct the boundaries in Advanced.')
    if view in FRONT_VIEWS and optical and not support:
        issue('support-unverified', 'bridge', 'Optical boundaries alone do not establish frame construction.',
              'Review a visible rim or rimless bridge attachment; do not infer rimless from missing rims.')
    if view in SIDE_VIEWS and not arm:
        issue('arm-unverified', 'temples', 'No continuous single-arm profile was verified.',
              'Provide a sharper side view with one hinge and tip visible, or correct the branch.')
    if view in FRONT_VIEWS:
        for side in ('left', 'right'):
            if f'nose-pad-{side}' not in indexed:
                issue('nose-support-unverified', f'nose-pad-{side}', 'Separate nose-pad geometry is not verified; this does not mean no pad exists.',
                      'Check a close inner/oblique photo. Integrated nose support does not require a separate pad.')
    edges = [dict(source=f'lens-{side}', relation='supported-by', target='front') for side in ('left', 'right')]
    if category == 'rimless':
        edges = [dict(source=f'lens-{side}', relation='attached-to', target='bridge-hardware') for side in ('left', 'right')]
    edges += [dict(source=f'temple-{side}', relation='attached-at', target=f'hinge-{side}') for side in ('left', 'right')]
    edges += [dict(source=f'nose-pad-{side}', relation='near', target='bridge') for side in ('left', 'right')]
    return dict(schema_version=SCHEMA_VERSION, scope='photo', photo_id=photo_id, view=view,
                status=status, method='geometric-anatomy-audit', nodes=nodes, relationships=edges,
                issues=issues, blocking=blocking, limitations=[
                    'Structural compatibility is not semantic recognition or a calibrated confidence probability.',
                    'Unobserved parts and material/depth cannot be recovered from this graph alone.'])


def attach_evidence(photo, result, width, height):
    """Keep diagnostic provenance attached to suggestions; never auto-approve."""
    report = audit_photo(photo, result.get('annotations', []),
                         result.get('construction', {}).get('category'), width=width, height=height)
    by_part = {n['id']: n for n in report['nodes']}
    for annotation in result.get('annotations', []):
        node = by_part.get(annotation['part'])
        annotation['evidence'] = dict(schema_version=SCHEMA_VERSION,
                                      state=node['state'] if node else 'uncertain',
                                      method='geometric-anatomy-audit',
                                      review_required=True,
                                      notes=[i['message'] for i in report['issues'] if i['part'] == annotation['part']])
    result['anatomy'] = report
    report['appearance_candidates'] = [dict(part=e.get('part'), photo_id=str(photo.get('id', '')),
                                            state='uncertain', role='inner-arm-visible-region',
                                            used_for_geometry=False, reason=e.get('reason', ''))
                                       for e in result.get('appearance_evidence', [])]
    return result


def audit_project(snapshot, active_annotations=None):
    """Aggregate active approved geometry only; do not register non-coplanar photos.

    Fitting views contribute structural evidence, not an assertion that every
    photo's pixels were applied. The existing multiview report covers photo use.
    """
    annotations = active_annotations if active_annotations is not None else snapshot.get('annotations', [])
    photos = [p for p in snapshot.get('photos', []) if p.get('role', 'fitting') != 'validation']
    ids = {str(p['id']) for p in photos}
    approved = [a for a in annotations if a.get('approved') and str(a.get('photo_id')) in ids]
    registration_issues=[]
    for a in approved:
        reg=a.get('registration')
        if not reg:continue
        source=next((p for p in photos if str(p['id'])==str(reg['source_photo_id'])),None)
        from .engine import _is_validation_photo
        if source is None or _is_validation_photo(source):
            registration_issues.append(dict(code='invalid-component-source',part=a['part'],message='Registered component needs an eligible source photo.',action='Reanalyze its fitting photos.',blocking=True))
        elif _polygon({'points':reg['source_points']},source.get('width',0),source.get('height',0))[0] is None:
            registration_issues.append(dict(code='invalid-component-source-outline',part=a['part'],message='Registered component source outline is invalid.',action='Reanalyze its fitting photos.',blocking=True))
    reports = [audit_photo(p, approved, snapshot.get('category')) for p in photos]
    nodes = []
    for part in PARTS:
        candidates = [n for r in reports for n in r['nodes'] if n['id'] == part and n['evidence']]
        state = ('observed' if any(n['state'] == 'observed' for n in candidates)
                 else 'inferred' if any(n['state'] == 'inferred' for n in candidates)
                 else 'uncertain')
        if part == 'front' and snapshot.get('category') == 'rimless' and not candidates:
            state = 'not-applicable'
        if part == 'shield' and snapshot.get('category') != 'sports-shield':
            state = 'not-applicable'
        if part.startswith('lens-') and snapshot.get('category') == 'sports-shield':
            state = 'not-applicable'
        nodes.append(dict(id=part, state=state, evidence=[e for n in candidates for e in n['evidence']]))
    warnings = []
    for side in ('left', 'right'):
        node = next(n for n in nodes if n['id'] == f'temple-{side}')
        if node['state'] == 'uncertain':
            warnings.append(f'{side} arm: no verified active side geometry; request that side photo or correct its outline.')
        elif node['state'] == 'inferred':
            warnings.append(f'{side} arm: explicit symmetry/shape inference, not independent photographic evidence.')
    # A reference-only inner/rear image is not proof of inner-arm texture.
    warnings.append('Inner-arm design requires visible, registered arm-surface evidence; a rear frame view alone is insufficient.')
    return dict(schema_version=SCHEMA_VERSION, scope='project', status='review-required',
                method='geometric-anatomy-audit', nodes=nodes,
                relationships=reports[0]['relationships'] if reports else [],
                photos=reports, issues=[i for r in reports for i in r['issues'] if i['blocking']]+registration_issues,
                blocking=bool(registration_issues) or any(r['blocking'] for r in reports), guidance=warnings,
                limitations=['Graph describes evidence, not calibrated hidden hardware or exact reconstruction.',
                             'Independent unseen-product evaluation is required before claiming production coverage.'])
