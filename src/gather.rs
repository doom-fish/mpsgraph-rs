use crate::checks;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::Tensor;
use core::ffi::c_char;
use core::ptr;
use std::ffi::CString;

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

const INDEX_TYPE: &str = "indices must be an integer tensor";

fn batch_prefix_matches(first: &[isize], second: &[isize], batch: usize) -> Result<()> {
    if checks::same_dims(&first[..batch], &second[..batch]) {
        Ok(())
    } else {
        Err(Error::InvalidShape(
            "updates and indices must match along the batch dimensions",
        ))
    }
}

pub(crate) fn check_gather_nd(
    updates: &[isize],
    indices: &[isize],
    batch: usize,
    result_rank_needed: bool,
) -> Result<()> {
    if batch >= updates.len() || batch >= indices.len() {
        return Err(Error::InvalidShape(
            "batch_dimensions must be less than the rank of every operand",
        ));
    }
    batch_prefix_matches(updates, indices, batch)?;
    let slice_rank = updates.len() - batch;
    let index_rank = indices.len() - batch;
    if let Some(depth) = indices.last().copied().and_then(checks::known) {
        if depth > slice_rank {
            return Err(Error::InvalidShape(
                "the index depth exceeds the updates' rank after the batch dimensions",
            ));
        }
        if result_rank_needed && slice_rank - depth + index_rank - 1 == 0 {
            return Err(Error::InvalidShape(
                "the gathered result would have rank 0",
            ));
        }
    }
    Ok(())
}

impl crate::graph::Graph {
/// Calls the `MPSGraph` framework counterpart for `gather_nd`.
    pub fn gather_nd(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[updates_tensor, indices_tensor])?;
        checks::integer_tensor(indices_tensor, INDEX_TYPE)?;
        if let (Some(updates), Some(indices)) = (updates_tensor.shape(), indices_tensor.shape()) {
            check_gather_nd(&updates, &indices, batch_dimensions, true)?;
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather_nd(
                self.as_ptr(),
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                batch_dimensions,
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[updates_tensor, indices_tensor],
            "MPSGraph did not create the gather",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `gather`.
    pub fn gather(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: usize,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[updates_tensor, indices_tensor])?;
        checks::integer_tensor(indices_tensor, INDEX_TYPE)?;
        if batch_dimensions > axis {
            return Err(Error::InvalidShape(
                "batch_dimensions must not exceed the gather axis",
            ));
        }
        if let (Some(updates), Some(indices)) = (updates_tensor.shape(), indices_tensor.shape()) {
            if axis >= updates.len() {
                return Err(Error::InvalidShape(
                    "the gather axis must be below the updates' rank",
                ));
            }
            if batch_dimensions > 0 {
                if batch_dimensions >= indices.len() {
                    return Err(Error::InvalidShape(
                        "batch_dimensions must be less than the rank of every operand",
                    ));
                }
                batch_prefix_matches(&updates, &indices, batch_dimensions)?;
            }
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather(
                self.as_ptr(),
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                axis,
                batch_dimensions,
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[updates_tensor, indices_tensor],
            "MPSGraph did not create the gather",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `gather_along_axis`.
    pub fn gather_along_axis(
        &self,
        axis: isize,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[updates_tensor, indices_tensor])?;
        checks::integer_tensor(indices_tensor, INDEX_TYPE)?;
        check_along_axis(updates_tensor, indices_tensor, Some(axis))?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather_along_axis(
                self.as_ptr(),
                axis,
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[updates_tensor, indices_tensor],
            "MPSGraph did not create the gather",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `gather_along_axis_tensor`.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn gather_along_axis_tensor(
        &self,
        axis_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[axis_tensor, updates_tensor, indices_tensor])?;
        checks::integer_tensor(indices_tensor, INDEX_TYPE)?;
        checks::index_scalar(axis_tensor, "the axis must be a rank-0 int32 or int64 tensor")?;
        check_along_axis(updates_tensor, indices_tensor, None)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather_along_axis_tensor(
                self.as_ptr(),
                axis_tensor.as_ptr(),
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[axis_tensor, updates_tensor, indices_tensor],
            "MPSGraph did not create the gather",
        )
    }
}

pub(crate) fn check_along_axis(
    values: &Tensor,
    indices: &Tensor,
    axis: Option<isize>,
) -> Result<()> {
    let (Some(values), Some(indices)) = (values.shape(), indices.shape()) else {
        return Ok(());
    };
    if values.is_empty() || values.len() != indices.len() {
        return Err(Error::InvalidShape(
            "the values and indices need the same rank of 1 or more",
        ));
    }
    let Some(axis) = axis else {
        return Ok(());
    };
    let axis = checks::normalized_axis(axis, values.len())
        .ok_or(Error::InvalidShape("the axis is outside -rank..rank"))?;
    let mismatched = values
        .iter()
        .zip(&indices)
        .enumerate()
        .any(|(index, (value, index_extent))| {
            index != axis && !checks::same_extent(*value, *index_extent)
        });
    if mismatched {
        Err(Error::InvalidShape(
            "the values and indices must match except along the axis",
        ))
    } else {
        Ok(())
    }
}
