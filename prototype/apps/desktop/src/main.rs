#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod paths;
mod state;

use std::sync::{Arc, Mutex};

use commands::{
    data_attribution, image_png, load_image, pick_image, recompute_overlay, solve_image,
    startup_request,
};
use state::AppState;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(Mutex::new(AppState::default())))
        .invoke_handler(tauri::generate_handler![
            pick_image,
            load_image,
            image_png,
            solve_image,
            recompute_overlay,
            startup_request,
            data_attribution,
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Starglyph");
}
