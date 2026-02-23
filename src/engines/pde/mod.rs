//! Module `engines::pde::mod`.
//!
//! Implements mod abstractions and re-exports used by adjacent pricing/model modules.
//!
//! References: Hull (11th ed.) Ch. 20, Tavella and Randall (2000), finite-difference stencils around Eq. (20.11)-(20.13).
//!
//! Primary API surface: module-level exports and submodule wiring.
//!
//! Numerical considerations: grid spacing, time-step size, and boundary handling dominate stability/consistency; check monotonicity and CFL-style limits where explicit terms appear.
//!
//! When to use: use PDE engines for early-exercise and barrier-style boundary problems in low dimensions; switch to Monte Carlo or trees for high-dimensional state spaces.

pub mod adi;
pub mod crank_nicolson;
pub mod explicit_fd;
mod fd_common;
pub mod hopscotch;
pub mod implicit_fd;

pub use adi::{AdiHestonEngine, AdiScheme};
pub use crank_nicolson::{BermudanPdeOutput, CrankNicolsonEngine, PdeExerciseBoundaryPoint};
pub use explicit_fd::ExplicitFdEngine;
pub use hopscotch::HopscotchEngine;
pub use implicit_fd::ImplicitFdEngine;
