pub mod american;
pub mod asian;
pub mod barrier;
pub mod bermudan;
pub mod european;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub fn sign(self) -> f64 {
        match self {
            Self::Call => 1.0,
            Self::Put => -1.0,
        }
    }
}
