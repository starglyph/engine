fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "pick_image",
            "load_image",
            "image_png",
            "solve_image",
            "startup_request",
            "data_attribution",
        ]),
    ))
    .expect("failed to build tauri application");
}
