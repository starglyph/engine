//! Observation-only per-FOV counters; installed only in the isolated harness.
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Funnel {
    pub fov_deg: f32,
    pub retained: usize,
    pub patterns: u64,
    pub hash_candidates: u64,
    pub fov_pass: u64,
    pub ratios_pass: u64,
    pub svd_pass: u64,
    pub probability_pass: u64,
    pub finalized: u64,
    pub verifications: u64,
    pub nonfinite_probability: u64,
    pub matches_histogram: BTreeMap<usize, u64>,
    pub best_ratio: Option<f64>,
    pub best: Option<serde_json::Value>,
}

impl Funnel {
    pub fn new(fov_deg: f32) -> Self {
        Self {
            fov_deg,
            retained: 0,
            patterns: 0,
            hash_candidates: 0,
            fov_pass: 0,
            ratios_pass: 0,
            svd_pass: 0,
            probability_pass: 0,
            finalized: 0,
            verifications: 0,
            nonfinite_probability: 0,
            matches_histogram: BTreeMap::new(),
            best_ratio: None,
            best: None,
        }
    }

    /// Record every verification, returning true only for a new minimum ratio.
    /// This return value controls diagnostic detail capture, never acceptance.
    pub fn observe(&mut self, probability: f64, threshold: f64, matches: usize) -> bool {
        self.verifications += 1;
        *self.matches_histogram.entry(matches).or_default() += 1;
        let ratio = probability / threshold;
        if !ratio.is_finite() || threshold <= 0.0 {
            self.nonfinite_probability += 1;
            return false;
        }
        if self.best_ratio.is_none_or(|previous| ratio < previous) {
            self.best_ratio = Some(ratio);
            true
        } else {
            false
        }
    }

    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({"fov_deg":self.fov_deg,"retained":self.retained,
            "patterns":self.patterns,"hash_candidates":self.hash_candidates,
            "fov_pass":self.fov_pass,"ratios_pass":self.ratios_pass,"svd_pass":self.svd_pass,
            "verifications":self.verifications,"probability_pass":self.probability_pass,
            "finalized":self.finalized,"nonfinite_probability":self.nonfinite_probability,
            "matches_histogram":self.matches_histogram,"best_ratio":self.best_ratio,"best":self.best})
    }
}

impl Drop for Funnel {
    fn drop(&mut self) {
        eprintln!(
            "SGTRACE {}",
            serde_json::json!({"stage":"funnel","data":self.snapshot()})
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_all_matches_and_keeps_strict_minimum() {
        let mut f = Funnel::new(22.0);
        assert!(f.observe(0.5, 0.1, 2));
        assert!(f.observe(0.2, 0.1, 4));
        assert!(!f.observe(0.3, 0.1, 4));
        assert!(!f.observe(0.2, 0.1, 4));
        assert_eq!(f.best_ratio, Some(2.0));
        assert_eq!(f.verifications, 4);
        assert_eq!(f.matches_histogram, BTreeMap::from([(2, 1), (4, 3)]));
        assert_eq!(f.probability_pass, 0); // Observation never decides acceptance.
    }

    #[test]
    fn nonfinite_diagnostics_do_not_panic_or_invent_a_best() {
        let mut f = Funnel::new(22.0);
        for (p, t) in [(f64::NAN, 0.1), (1.0, 0.0), (1.0, -1.0)] {
            assert!(!f.observe(p, t, 0));
        }
        assert_eq!(f.nonfinite_probability, 3);
        assert_eq!(f.verifications, 3);
        assert_eq!(f.best_ratio, None);
        assert!(f.snapshot()["best_ratio"].is_null());
    }
}
