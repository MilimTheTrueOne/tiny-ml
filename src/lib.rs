#![feature(portable_simd)]

mod networks;
mod training;

pub mod prelude {
    pub use super::networks::*;
    pub use super::training::*;
}
