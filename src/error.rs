/// Mirrors the `MPSGraph` framework counterpart for `Result`.
pub type Result<T> = core::result::Result<T, Error>;

/// Mirrors the `MPSGraph` framework counterpart for `Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
/// Mirrors the `MPSGraph` framework case `InvalidLength`.
    InvalidLength { expected: usize, actual: usize },
/// Mirrors the `MPSGraph` framework case `OperationFailed`.
    OperationFailed(&'static str),
/// Mirrors the `MPSGraph` framework case `UnsupportedDataType`.
    UnsupportedDataType(u32),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(
                    f,
                    "invalid buffer length: expected {expected} bytes, got {actual}"
                )
            }
            Self::OperationFailed(message) => f.write_str(message),
            Self::UnsupportedDataType(data_type) => {
                write!(f, "unsupported MPSDataType raw value: {data_type:#x}")
            }
        }
    }
}

impl std::error::Error for Error {}
