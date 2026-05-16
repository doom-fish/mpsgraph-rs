pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidLength { expected: usize, actual: usize },
    OperationFailed(&'static str),
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
