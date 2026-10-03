// Research-only entry point appended to solver/solve.rs in the fixture build.
// Exercise the actual verify_attitude, not a reimplementation of clipping.
pub fn research_footprint_fixture(
    database: &str,
    portrait: bool,
    distractors: bool,
    edge: bool,
    refined_ratio: f32,
) -> crate::Result<usize> {
    let mut db = SolverDatabase::load_from_file(database)?;
    let (width, height) = (1000, if portrait { 1800 } else { 600 });
    let fov = 40.0_f32.to_radians();
    let half_width = (fov / 2.0).tan();
    let half_height = half_width * height as f32 / width as f32;
    let radius = 0.01 * fov;
    let unit = |x: f32, y: f32| {
        let norm = (1.0 + x*x + y*y).sqrt();
        [x/norm, y/norm, 1.0/norm]
    };
    let mut image_vectors = Vec::new();
    let mut catalog_cam = Vec::new();
    if distractors {
        // Brighter than every image star, within the cone but outside the sensor.
        for i in 0..32 {
            catalog_cam.push(unit(half_width + 0.05 + i as f32*0.001, 0.0));
        }
    }
    for i in 0..12 {
        let x = (-0.75 + (i%4) as f32*0.5) * half_width;
        let y = (-0.6 + (i/4) as f32*0.6) * half_height;
        let (mut cx, mut cy, mut ix, mut iy) = (x,y,x,y);
        if edge && i == 0 {
            if portrait {
                // Mid-edge stays inside the unchanged circular query even
                // when refined FOV is smaller than the centroid-vector FOV.
                cx = 0.0;
                ix = 0.0;
                cy = half_height + 0.4*radius;
                iy = half_height - 0.1*radius;
            } else {
                cy = 0.0;
                iy = 0.0;
                cx = half_width + 0.4*radius;
                ix = half_width - 0.1*radius;
            }
        }
        image_vectors.push(unit(ix, iy));
        catalog_cam.push(unit(cx, cy));
    }
    // Camera boresight at ICRS RA=0, Dec=0, away from polar query singularities.
    let stars: Vec<_> = catalog_cam.iter().enumerate().map(|(i,v)| crate::Star {
        id: i as i64 + 1, ra_rad: v[0].atan2(v[2]), dec_rad: v[1].asin(), mag:i as f32/10.,
    }).collect();
    db.star_catalog_ids = stars.iter().map(|s| s.id).collect();
    db.star_vectors = stars.iter().map(|s| {
        let v = s.uvec(); [v[0],v[1],v[2]]
    }).collect();
    db.star_catalog = crate::StarCatalog::new(16, stars);
    let mut rotation = Matrix3::zeros();
    rotation[(0,1)]=1.0; rotation[(1,2)]=1.0; rotation[(2,0)]=1.0;
    let config = SolveConfig {
        match_radius: 0.01,
        ..SolveConfig::new(fov,width,height)
    };
    let pixel_scale = 2.0 * half_width / width as f32;
    let (matches, _) = db.verify_attitude(&rotation, &image_vectors, image_vectors.len(), fov * refined_ratio, &config, /*PIXEL_SCALE_ARG*/ &db.star_vectors);
    let _ = pixel_scale;
    Ok(matches.len())
}
