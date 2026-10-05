//! Diagnostic-only observation of real tetra3 solutions before acceptance/refine.
use super::*;
use serde_json::{json, Value};

fn camera(p: &CameraSolution) -> Value {
    json!({"ra_deg":p.ra_deg,"dec_deg":p.dec_deg,"roll_deg":p.roll_deg,
        "focal_px":p.focal_px,"width":p.width,"height":p.height,"k1":p.k1})
}

pub(super) fn emit(
    sol: &Solution,
    detections: &[Detection],
    stars: &VerifyStars,
    cand: &Candidate,
) {
    let rot = cand.pose.rotation();
    let rot0 = geom::pose_to_rotation(cand.pose.ra_deg, cand.pose.dec_deg, 0.0);
    let (cx, cy) = cand.pose.principal_point();
    let pairs: Vec<_> = sol
        .matched_centroid_indices
        .iter()
        .enumerate()
        .map(|(i, &index)| {
            let id = sol.matched_catalog_ids[i];
            let world = stars.by_id.get(&id);
            let det = detections.get(index);
            let mut row = json!({"catalog_id":id,"detection_index":index});
            if let (Some(world), Some(det)) = (world, det) {
                let (ra, dec) = geom::unit_to_radec(*world);
                let native = sol.world_to_pixel(ra, dec);
                let input = detection_to_centroid(det, sol.image_width, sol.image_height);
                let projected = geom::project(
                    &rot,
                    cand.pose.focal_px,
                    0.,
                    sol.image_width,
                    sol.image_height,
                    *world,
                );
                let cam = rot0 * Vector3::from_column_slice(world);
                let roll = (cam.y / cam.z).atan2(cam.x / cam.z) - (cy - det.y).atan2(det.x - cx);
                row["world"] = json!(world);
                row["radec"] = json!([ra, dec]);
                row["xy"] = json!([det.x, det.y]);
                row["input_centroid"] = json!([input.x, input.y]);
                row["tetra_centered_xy"] = json!(native);
                row["adapter_xy"] = json!(projected);
                row["pair_roll_deg"] = json!(roll.to_degrees());
            }
            row
        })
        .collect();
    let verified: Vec<_> = cand
        .verify
        .matches
        .iter()
        .zip(&cand.verify.matched_detections)
        .map(|(m, i)| json!({"world":m.world,"xy":[m.px,m.py],"detection_index":i}))
        .collect();
    // Deliberately only observe: no mutation of candidate, input or acceptance.
    eprintln!(
        "CANDIDATE_TRACE {}",
        json!({"solution":sol,"camera":camera(&cand.pose),"label":cand.label,
        "detections":detections.iter().map(|d|json!([d.x,d.y,d.flux])).collect::<Vec<_>>(),
        "pairs":pairs,"verification":{"hits":cand.verify.hits,"predicted":cand.verify.predicted,
            "log_odds":cand.verify.log_odds,"rms_px":cand.verify.rms_px,"matches":verified}})
    );
}
