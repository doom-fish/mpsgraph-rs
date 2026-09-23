/// Mirrors the `MPSGraph` framework counterpart for `Result`.
pub type Result<T> = core::result::Result<T, Error>;

/// Mirrors the `MPSGraph` framework counterpart for `Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
/// Mirrors the `MPSGraph` framework case `InvalidLength`.
    InvalidLength { expected: usize, actual: usize },
/// Mirrors the `MPSGraph` framework case `OperationFailed`.
    OperationFailed(&'static str),
/// Mirrors the `MPSGraph` framework case `UnsupportedDataType`.
    UnsupportedDataType(u32),
    BufferTooSmall {
        required: usize,
        length: usize,
    },
    Overflow,
    InvalidShape(&'static str),
    ExecutionFailed(String),
    Unsupported(&'static str),
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
            Self::BufferTooSmall { required, length } => write!(
                f,
                "Metal buffer too small: {required} bytes required, buffer has {length}"
            ),
            Self::Overflow => f.write_str("size computation overflowed"),
            Self::InvalidShape(message) => write!(f, "invalid shape or axis: {message}"),
            Self::ExecutionFailed(message) => write!(f, "graph execution failed: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
        }
    }
}

impl std::error::Error for Error {}
