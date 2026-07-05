pub mod benchmark;
pub mod config;
pub mod contracts;
pub mod detection;
pub mod io;
pub mod matching;
pub mod overlay;
pub mod pipeline;
pub mod pose;

pub use benchmark::{run_benchmark, BenchmarkConfig, BenchmarkReport};
pub use config::SolverConfig;
pub use contracts::{SolveStatus, SolverResult};
pub use pipeline::{solve_frame, solve_frame_with_outputs};
