use anyhow::{bail, Result};
use clap::Parser;
use std::path::PathBuf;

use simulator_core::config::{
    CameraConfig, CameraSamplingConfig, CatalogConfig, DatasetConfig, DegradationConfig,
    RenderConfig, SplitConfig,
};

#[derive(Debug, Parser)]
#[command(name = "dataset-cli")]
#[command(about = "Starglyph phase-1 dataset generator prototype")]
struct Args {
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "artifacts/simulator/dataset-v1")]
    output_root: PathBuf,
    #[arg(long, default_value_t = 100)]
    train_frames: usize,
    #[arg(long, default_value_t = 20)]
    val_frames: usize,
    #[arg(long, default_value_t = 20)]
    test_frames: usize,
    #[arg(long, default_value = "../data/catalogs/hyg_v3.csv")]
    catalog_csv: PathBuf,
    #[arg(long, default_value_t = false)]
    use_baseline_catalog: bool,
    #[arg(long, default_value_t = false)]
    validate_reproducibility: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.use_baseline_catalog && !args.catalog_csv.exists() {
        bail!(
            "catalog csv '{}' was not found. run `make fetch-catalog` (from `prototype/`) or pass `--use-baseline-catalog`",
            args.catalog_csv.display()
        );
    }

    let config = DatasetConfig {
        seed: args.seed,
        output_root: args.output_root,
        splits: SplitConfig {
            train_frames: args.train_frames,
            val_frames: args.val_frames,
            test_frames: args.test_frames,
        },
        catalog: CatalogConfig {
            csv_path: if args.use_baseline_catalog {
                None
            } else {
                Some(args.catalog_csv)
            },
            name: if args.use_baseline_catalog {
                "baseline-built-in".to_string()
            } else {
                "hyg-v3".to_string()
            },
            subset: if args.use_baseline_catalog {
                "bright-stars-sample".to_string()
            } else {
                "full".to_string()
            },
            license: if args.use_baseline_catalog {
                "internal-demo".to_string()
            } else {
                "CC BY-SA 4.0".to_string()
            },
        },
        camera: CameraConfig::default(),
        camera_sampling: CameraSamplingConfig::default(),
        render: RenderConfig::default(),
        degradations: DegradationConfig::default(),
    };

    let manifest = simulator_core::generate_dataset(&config)?;
    let output = serde_json::to_string_pretty(&manifest)?;
    println!("{output}");

    if args.validate_reproducibility {
        let report = simulator_core::validate_reproducibility(&config)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(())
}
