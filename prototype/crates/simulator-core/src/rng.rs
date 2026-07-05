use std::fs;
use std::path::{Path, PathBuf};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

use crate::dataset::DatasetSplit;

#[derive(Debug, Clone, Copy)]
pub struct SeedDeriver {
    global_seed: u64,
}

impl SeedDeriver {
    #[must_use]
    pub fn new(global_seed: u64) -> Self {
        Self { global_seed }
    }

    #[must_use]
    pub fn frame_seed(&self, split: DatasetSplit, frame_index: usize) -> u64 {
        let split_mix = match split {
            DatasetSplit::Train => 0xA8D0_5B9D_3A2F_8C01,
            DatasetSplit::Val => 0x17C5_841A_72B4_41ED,
            DatasetSplit::Test => 0xE7F1_AA93_81CC_5D2B,
        };
        splitmix64(self.global_seed ^ split_mix ^ (frame_index as u64).wrapping_mul(0x9E37_79B9))
    }

    #[must_use]
    pub fn frame_rng(&self, split: DatasetSplit, frame_index: usize) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(self.frame_seed(split, frame_index))
    }
}

#[must_use]
pub fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn directory_digest(path: &Path) -> std::io::Result<String> {
    let mut files = Vec::new();
    collect_files(path, path, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    for (relative, absolute) in files {
        hasher.update(relative.as_os_str().as_encoded_bytes());
        hasher.update(fs::read(absolute)?);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_files(
    root: &Path,
    current: &Path,
    out: &mut Vec<(PathBuf, PathBuf)>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .map(Path::to_path_buf)
            .unwrap_or_else(|_| path.clone());
        out.push((relative, path));
    }
    Ok(())
}
