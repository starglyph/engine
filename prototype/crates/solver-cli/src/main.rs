use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use simulator_core::config::CatalogConfig;
use solver_core::{run_benchmark, BenchmarkConfig, SolverConfig};

#[derive(Debug, Parser)]
#[command(name = "solver-cli")]
#[command(about = "Starglyph phase-2 solver benchmark runner")]
struct Args {
    #[arg(long, default_value = "artifacts/simulator/dataset-v1")]
    dataset_root: PathBuf,
    #[arg(long, default_value = "artifacts/recognizer/run-latest")]
    output_root: PathBuf,
    #[arg(long, value_delimiter = ',')]
    split: Vec<String>,
    #[arg(long, default_value_t = false)]
    skip_reproducibility_check: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut config = SolverConfig::default();
    config.benchmark.run_reproducibility_check = !args.skip_reproducibility_check;
    let benchmark_config = BenchmarkConfig {
        dataset_root: args.dataset_root,
        output_root: args.output_root,
        split_filter: if args.split.is_empty() {
            None
        } else {
            Some(args.split)
        },
        solver: config,
        catalog: CatalogConfig::default(),
    };
    let report = run_benchmark(&benchmark_config)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
