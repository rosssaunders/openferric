//! Finite-difference PDE pricing engines.

pub mod crank_nicolson;
pub mod explicit_fd;
pub mod hopscotch;
pub mod implicit_fd;

pub use crank_nicolson::CrankNicolsonEngine;
pub use explicit_fd::ExplicitFdEngine;
pub use hopscotch::HopscotchEngine;
pub use implicit_fd::ImplicitFdEngine;
