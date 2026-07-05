use anyhow::Result;
use simulator_core::config::{
    CameraConfig, CameraSamplingConfig, CatalogConfig, DatasetConfig, DegradationConfig,
    RenderConfig, SplitConfig,
};
use solver_core::{run_benchmark, BenchmarkConfig, SolverConfig};

#[test]
fn benchmark_is_reproducible_for_fixed_dataset_and_config() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let dataset_root = temp.path().join("dataset");
    let output_root = temp.path().join("recognizer-run");

    let dataset_config = DatasetConfig {
        seed: 42,
        output_root: dataset_root.clone(),
        splits: SplitConfig {
            train_frames: 2,
            val_frames: 1,
            test_frames: 1,
        },
        catalog: CatalogConfig::default(),
        camera: CameraConfig::default(),
        camera_sampling: CameraSamplingConfig::default(),
        render: RenderConfig::default(),
        degradations: DegradationConfig::default(),
    };
    simulator_core::generate_dataset(&dataset_config)?;

    let mut solver = SolverConfig::default();
    solver.benchmark.run_reproducibility_check = true;
    solver.benchmark.worst_case_count = 2;
    let benchmark_config = BenchmarkConfig {
        dataset_root,
        output_root,
        split_filter: Some(vec!["test".to_string()]),
        solver,
        catalog: CatalogConfig::default(),
    };
    let report = run_benchmark(&benchmark_config)?;
    let repro = report
        .reproducibility
        .expect("reproducibility check should be populated");
    assert!(repro.passed, "benchmark reproducibility check should pass");
    Ok(())
}
