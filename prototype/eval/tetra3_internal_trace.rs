//! Observation-only hooks installed in an isolated research copy of tetra3.
use std::cell::{Cell, RefCell};
thread_local! {
    static IDS: RefCell<Vec<Option<i64>>> = const { RefCell::new(Vec::new()) };
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}
pub fn set_ids(ids: Vec<Option<i64>>) { IDS.with(|v| *v.borrow_mut() = ids); }
pub fn set_active(value: bool) { ACTIVE.with(|v| v.set(value)); }
pub fn active() -> bool { ACTIVE.with(|v| v.get()) }
pub fn catalog_is_target(mut ids: [i64; 4]) -> bool {
    ids.sort_unstable();
    ids == [73486, 74164, 76644, 78029]
}
pub fn image_is_target(indices: [usize; 4]) -> bool {
    IDS.with(|v| {
        let ids = v.borrow();
        catalog_is_target(indices.map(|i| ids.get(i).copied().flatten().unwrap_or(-1)))
    })
}
pub fn emit(stage: &str, data: serde_json::Value) {
    eprintln!("SGTRACE {}", serde_json::json!({"stage":stage, "data":data}));
}
