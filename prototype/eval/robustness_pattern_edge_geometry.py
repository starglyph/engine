"""Identified-edge diagnostics in f64, cross-checked against saved tetra3 f32."""
from itertools import combinations

import numpy as np

PAIRS = tuple(combinations(range(4), 2))


def vectors(xy, focal):
    xy = np.asarray(xy, float)
    if xy.shape != (4, 2) or not np.isfinite(xy).all() or not np.isfinite(focal) or focal <= 0:
        raise ValueError('four finite positions and positive focal required')
    v = np.column_stack((xy / focal, np.ones(4)))
    return v / np.linalg.norm(v, axis=1)[:, None]


def angles(v):
    v = np.asarray(v, float)
    if v.shape != (4, 3) or not np.isfinite(v).all():
        raise ValueError('four finite rays required')
    return np.array([2*np.arcsin(np.clip(np.linalg.norm(v[i]-v[j])/2, 0, 1)) for i,j in PAIRS])


def signature(xy, focal):
    edge = angles(vectors(xy, focal))
    order = np.argsort(edge, kind='stable')
    if edge[order[-1]] <= 0:
        raise ValueError('nonzero pattern span required')
    return edge, order, edge[order[:5]] / edge[order[-1]]


def identified_ratios(xy, focal, order):
    edge = angles(vectors(xy, focal))
    return edge[order[:5]] / edge[order[-1]]


def ratio_terms(image, catalog, order):
    """Exact numerator/denominator decomposition with catalogue edge identities."""
    image, catalog = np.asarray(image), np.asarray(catalog)
    denominator = order[-1]; numerators = order[:5]
    if min(image[denominator], catalog[denominator]) <= 0:
        raise ValueError('positive reference edge required')
    numerator = (image[numerators]-catalog[numerators])/image[denominator]
    scale = -catalog[numerators]*(image[denominator]-catalog[denominator])/(image[denominator]*catalog[denominator])
    return numerator, scale


def jacobian(xy, focal, order, step):
    if step <= 0:raise ValueError('positive derivative step required')
    jac = np.empty((5,4,2))
    for i in range(4):
        for axis in range(2):
            delta = np.zeros((4,2)); delta[i,axis] = step
            jac[:,i,axis] = (identified_ratios(xy+delta,focal,order)-identified_ratios(xy-delta,focal,order))/(2*step)
    return jac


def attribute(ideal, measured, focal, order):
    ideal, measured = np.asarray(ideal), np.asarray(measured)
    displacement = measured-ideal
    jac = jacobian(ideal,focal,order,1e-3)
    alternate = jacobian(ideal,focal,order,5e-4)
    radius = np.linalg.norm(ideal,axis=1)
    radial_unit = np.divide(ideal,radius[:,None],out=np.zeros_like(ideal),where=radius[:,None]>1e-12)
    radial = radial_unit*np.sum(displacement*radial_unit,axis=1)[:,None]
    transverse = displacement-radial
    contributions = np.sum(jac*displacement[None,:,:],axis=2)
    radial_contributions = np.sum(jac*radial[None,:,:],axis=2)
    transverse_contributions = np.sum(jac*transverse[None,:,:],axis=2)
    actual = identified_ratios(measured,focal,order)-identified_ratios(ideal,focal,order)
    return dict(displacement_xy_px=displacement.tolist(),radial_displacement_xy_px=radial.tolist(),
        transverse_displacement_xy_px=transverse.tolist(),per_source=contributions.tolist(),
        radial_per_source=radial_contributions.tolist(),transverse_per_source=transverse_contributions.tolist(),
        actual_change=actual.tolist(),linear_change=contributions.sum(axis=1).tolist(),
        max_linear_error=float(np.max(np.abs(actual-contributions.sum(axis=1)))),
        max_step_change=float(np.max(np.abs(np.sum((jac-alternate)*displacement[None,:,:],axis=(1,2))))))
