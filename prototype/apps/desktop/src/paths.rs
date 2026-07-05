use std::path::{Path, PathBuf};

use serde::Serialize;

/// Resolved on-disk locations for catalog and constellation data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPaths {
    pub catalog: PathBuf,
    pub constellation_lines: PathBuf,
    pub constellation_names: PathBuf,
}

const CATALOG_CANDIDATES: &[&str] = &[
    "hyg_v42.csv.gz",
    "hyg_v42.csv",
    "hyg_v3.csv.gz",
    "hyg_v3.csv",
];

/// Locate Starglyph data files using env, filesystem walk, and optional Tauri resources.
pub fn resolve_data_paths(resource_dir: Option<&Path>) -> Result<DataPaths, String> {
    let mut attempted = Vec::new();

    if let Ok(dir) = std::env::var("STARGLYPH_DATA_DIR") {
        let root = PathBuf::from(&dir);
        attempted.push(format!("STARGLYPH_DATA_DIR={}", root.display()));
        if let Some(paths) = try_data_root(&root) {
            return Ok(paths);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            attempted.push(format!("executable dir {}", parent.display()));
            if let Some(root) = walk_up_for_data_root(parent) {
                if let Some(paths) = try_data_root(&root) {
                    return Ok(paths);
                }
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        attempted.push(format!("cwd {}", cwd.display()));
        if let Some(root) = walk_up_for_data_root(&cwd) {
            if let Some(paths) = try_data_root(&root) {
                return Ok(paths);
            }
        }
    }

    if let Some(resource) = resource_dir {
        attempted.push(format!("resource dir {}", resource.display()));
        if let Some(paths) = try_data_root(resource) {
            return Ok(paths);
        }
    }

    Err(format!(
        "could not locate Starglyph data files (catalog + constellations); searched: {}",
        attempted.join("; ")
    ))
}

fn walk_up_for_data_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if try_data_root(&current).is_some() {
            return Some(current);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

fn try_data_root(root: &Path) -> Option<DataPaths> {
    let catalogs_dir = root.join("data/catalogs");
    let celestial_dir = root.join("data/celestial");

    let catalog = CATALOG_CANDIDATES
        .iter()
        .map(|name| catalogs_dir.join(name))
        .find(|path| path.is_file())?;

    let lines = celestial_dir.join("constellations.lines.json");
    let names = celestial_dir.join("constellations.json");
    if !lines.is_file() || !names.is_file() {
        return None;
    }

    Some(DataPaths {
        catalog,
        constellation_lines: lines,
        constellation_names: names,
    })
}

/// Parse CLI args: first non-flag positional = image path; `--auto-solve` enables auto solve.
pub fn parse_startup_args(
    args: impl IntoIterator<Item = impl AsRef<str>>,
) -> Option<StartupRequest> {
    let mut image_path = None;
    let mut auto_solve = false;

    for arg in args {
        let arg = arg.as_ref();
        if arg == "--auto-solve" {
            auto_solve = true;
        } else if !arg.starts_with('-') && image_path.is_none() {
            image_path = Some(arg.to_string());
        }
    }

    image_path.map(|path| StartupRequest { path, auto_solve })
}

/// Startup image path and optional auto-solve flag from process args.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StartupRequest {
    pub path: String,
    pub auto_solve: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_layout(root: &Path) {
        let catalogs = root.join("data/catalogs");
        let celestial = root.join("data/celestial");
        fs::create_dir_all(&catalogs).unwrap();
        fs::create_dir_all(&celestial).unwrap();
        fs::write(catalogs.join("hyg_v3.csv"), "id,ra,dec,mag\n").unwrap();
        fs::write(celestial.join("constellations.lines.json"), "[]").unwrap();
        fs::write(celestial.join("constellations.json"), "{}").unwrap();
    }

    #[test]
    fn resolves_from_explicit_root() {
        let tmp = TempDir::new().unwrap();
        write_layout(tmp.path());

        let paths = try_data_root(tmp.path()).expect("layout");
        assert!(paths.catalog.ends_with("hyg_v3.csv"));
        assert!(paths
            .constellation_lines
            .ends_with("constellations.lines.json"));
    }

    #[test]
    fn resolves_via_walk_up() {
        let tmp = TempDir::new().unwrap();
        write_layout(tmp.path());
        let nested = tmp.path().join("prototype/apps/desktop");
        fs::create_dir_all(&nested).unwrap();

        let root = walk_up_for_data_root(&nested).expect("walk");
        assert_eq!(root, tmp.path());
    }

    #[test]
    fn parse_startup_args_extracts_path_and_flag() {
        let req = parse_startup_args(["--auto-solve", "/tmp/frame.png"]).expect("request");
        assert_eq!(req.path, "/tmp/frame.png");
        assert!(req.auto_solve);
    }

    #[test]
    fn parse_startup_args_ignores_flags_without_path() {
        assert!(parse_startup_args(["--auto-solve"]).is_none());
    }
}
