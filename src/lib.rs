mod networks;

pub mod prelude {
    pub use super::networks::*;
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2, 2)
    }
}
