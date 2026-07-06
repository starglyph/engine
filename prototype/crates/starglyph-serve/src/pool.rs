//! Pool of warmed [`Engine`]s, checked out one per in-flight solve.
//!
//! `solve_frame_with_engine` takes `&mut Engine` — a solve may lazily load or
//! build a dense-band database and append it to the engine — so engines cannot
//! be shared by reference across concurrent requests. One shared engine behind
//! a mutex would serialize every solve; a fresh engine per request would
//! re-deserialize multi-hundred-MB databases each time (the desktop app's
//! per-call `Engine::default()` anti-pattern). The pool instead hands each
//! request exclusive ownership of one warmed engine; the semaphore doubles as
//! the concurrent-solve limit.

use std::sync::{Arc, Mutex, PoisonError};

use starglyph_core::engine::Engine;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct EnginePool {
    semaphore: Arc<Semaphore>,
    idle: Arc<Mutex<Vec<Engine>>>,
}

impl EnginePool {
    /// Empty pool: checkouts wait until warmed engines are [`install`]ed.
    ///
    /// [`install`]: EnginePool::install
    #[must_use]
    pub fn new() -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(0)),
            idle: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a warmed engine and its permit (startup warmup path).
    pub fn install(&self, engine: Engine) {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(engine);
        self.semaphore.add_permits(1);
    }

    /// Wait for an idle engine and take exclusive ownership. Both the permit
    /// and an engine (the same one, or a fresh replacement if the solve
    /// panicked) must come back via [`EnginePool::checkin`].
    pub async fn checkout(&self) -> (OwnedSemaphorePermit, Engine) {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("pool semaphore is never closed");
        let engine = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .expect("a permit always implies an idle engine");
        (permit, engine)
    }

    /// Return an engine; the permit is released only after the engine is back
    /// in the idle list, so a woken waiter always finds one.
    pub fn checkin(&self, permit: OwnedSemaphorePermit, engine: Engine) {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(engine);
        drop(permit);
    }
}

impl Default for EnginePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn checkout_blocks_at_capacity_and_resumes_on_checkin() {
        let pool = EnginePool::new();
        pool.install(Engine::default());
        pool.install(Engine::default());

        let (permit_a, engine_a) = pool.checkout().await;
        let (_permit_b, _engine_b) = pool.checkout().await;

        // Pool exhausted: a third checkout must wait.
        assert!(
            timeout(Duration::from_millis(50), pool.checkout())
                .await
                .is_err(),
            "checkout should block when all engines are out"
        );

        pool.checkin(permit_a, engine_a);
        assert!(
            timeout(Duration::from_millis(50), pool.checkout())
                .await
                .is_ok(),
            "checkout should resume after a checkin"
        );
    }

    #[tokio::test]
    async fn empty_pool_blocks_until_first_install() {
        let pool = EnginePool::new();
        assert!(timeout(Duration::from_millis(50), pool.checkout())
            .await
            .is_err());
        pool.install(Engine::default());
        assert!(timeout(Duration::from_millis(50), pool.checkout())
            .await
            .is_ok());
    }
}
