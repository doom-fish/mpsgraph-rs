use crate::error::{Error, Result};
use crate::graph::{data_type, Tensor};

pub type Dims = Vec<isize>;

pub const fn is_float(data_type: u32) -> bool {
    matches!(
        data_type,
        data_type::FLOAT16 | data_type::FLOAT32 | data_type::BFLOAT16
    )
}

pub const fn is_complex(data_type: u32) -> bool {
    matches!(
        data_type,
        data_type::COMPLEX_FLOAT16 | data_type::COMPLEX_FLOAT32
    )
}

pub const fn is_integer(data_type: u32) -> bool {
    matches!(
        data_type,
        data_type::INT8
            | data_type::INT16
            | data_type::INT32
            | data_type::INT64
            | data_type::UINT8
            | data_type::UINT16
            | data_type::UINT32
            | data_type::UINT64
    )
}

pub const fn is_index(data_type: u32) -> bool {
    matches!(data_type, data_type::INT32 | data_type::INT64)
}

pub fn known(extent: isize) -> Option<usize> {
    usize::try_from(extent).ok()
}

pub const fn same_extent(first: isize, second: isize) -> bool {
    first < 0 || second < 0 || first == second
}

pub fn same_dims(first: &[isize], second: &[isize]) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second)
            .all(|(first, second)| same_extent(*first, *second))
}

pub fn broadcastable(first: &[isize], second: &[isize]) -> bool {
    first
        .iter()
        .rev()
        .zip(second.iter().rev())
        .all(|(first, second)| same_extent(*first, *second) || *first == 1 || *second == 1)
}

pub fn element_count(dims: &[isize]) -> Option<usize> {
    dims.iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(known(*extent)?))
}

pub fn normalized_axis(axis: isize, rank: usize) -> Option<usize> {
    let signed_rank = isize::try_from(rank).ok()?;
    let axis = if axis < 0 {
        axis.checked_add(signed_rank)?
    } else {
        axis
    };
    usize::try_from(axis).ok().filter(|axis| *axis < rank)
}

pub fn axis(tensor: &Tensor, axis: isize) -> Result<Option<(usize, Dims)>> {
    let Some(dims) = tensor.shape() else {
        return Ok(None);
    };
    let index = normalized_axis(axis, dims.len())
        .ok_or(Error::InvalidShape("the axis is outside -rank..rank"))?;
    Ok(Some((index, dims)))
}

pub fn rank_at_least(tensor: &Tensor, rank: usize, message: &'static str) -> Result<Option<Dims>> {
    match tensor.shape() {
        Some(dims) if dims.len() < rank => Err(Error::InvalidShape(message)),
        dims => Ok(dims),
    }
}

pub fn rank_exactly(tensor: &Tensor, rank: usize, message: &'static str) -> Result<Option<Dims>> {
    match tensor.shape() {
        Some(dims) if dims.len() != rank => Err(Error::InvalidShape(message)),
        dims => Ok(dims),
    }
}

pub fn dims_match(tensor: &Tensor, expected: &[isize], message: &'static str) -> Result<()> {
    match tensor.shape() {
        Some(dims) if !same_dims(&dims, expected) => Err(Error::InvalidShape(message)),
        _ => Ok(()),
    }
}

pub fn same_data_type(first: &Tensor, second: &Tensor, message: &'static str) -> Result<()> {
    if first.data_type() == second.data_type() {
        Ok(())
    } else {
        Err(Error::InvalidDataType(message))
    }
}

pub fn broadcast_with(first: &Tensor, second: &Tensor, message: &'static str) -> Result<()> {
    match (first.shape(), second.shape()) {
        (Some(first), Some(second)) if !broadcastable(&first, &second) => {
            Err(Error::InvalidShape(message))
        }
        _ => Ok(()),
    }
}

pub fn elementwise(first: &Tensor, second: &Tensor) -> Result<()> {
    same_data_type(first, second, "the operands need the same data type")?;
    broadcast_with(
        first,
        second,
        "the operand shapes are not broadcast compatible",
    )
}

pub fn float_tensor(tensor: &Tensor, message: &'static str) -> Result<()> {
    if is_float(tensor.data_type()) {
        Ok(())
    } else {
        Err(Error::InvalidDataType(message))
    }
}

pub fn integer_tensor(tensor: &Tensor, message: &'static str) -> Result<()> {
    if is_integer(tensor.data_type()) {
        Ok(())
    } else {
        Err(Error::InvalidDataType(message))
    }
}

pub fn index_scalar(tensor: &Tensor, message: &'static str) -> Result<()> {
    if !is_index(tensor.data_type()) {
        return Err(Error::InvalidDataType(message));
    }
    rank_exactly(tensor, 0, message).map(drop)
}

pub fn index_vector(tensor: &Tensor, length: Option<usize>, message: &'static str) -> Result<()> {
    if !is_index(tensor.data_type()) {
        return Err(Error::InvalidDataType(message));
    }
    let Some(dims) = rank_exactly(tensor, 1, message)? else {
        return Ok(());
    };
    match (length, known(dims[0])) {
        (Some(expected), Some(actual)) if expected != actual => Err(Error::InvalidShape(message)),
        _ => Ok(()),
    }
}

pub fn single_element(tensor: &Tensor, message: &'static str) -> Result<()> {
    match tensor.shape().as_deref().and_then(element_count) {
        Some(count) if count != 1 => Err(Error::InvalidShape(message)),
        _ => Ok(()),
    }
}

pub fn window_fits(
    extent: isize,
    kernel: usize,
    dilation: usize,
    padding: (usize, usize),
) -> bool {
    let Some(extent) = known(extent) else {
        return true;
    };
    let span = kernel
        .checked_sub(1)
        .and_then(|taps| taps.checked_mul(dilation))
        .and_then(|reach| reach.checked_add(1));
    let padded = extent
        .checked_add(padding.0)
        .and_then(|padded| padded.checked_add(padding.1));
    matches!((span, padded), (Some(span), Some(padded)) if span <= padded)
}

pub fn shape_to_dims(shape: &[usize]) -> Result<Dims> {
    shape
        .iter()
        .map(|extent| isize::try_from(*extent).map_err(|_| Error::Overflow))
        .collect()
}
