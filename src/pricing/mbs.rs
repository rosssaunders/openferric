//! MBS pricing — re-exports from instruments::mbs.
//!
//! The core pricing logic lives in [`crate::instruments::mbs`] alongside the
//! instrument definitions for cohesion. This module re-exports the key types
//! so they are accessible from the `pricing` namespace as well.

pub use crate::instruments::mbs::{
    ConstantCpr, IoStrip, MbsCashflow, MbsPassThrough, PoStrip, PrepaymentModel, PsaModel,
};
