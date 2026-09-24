use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{data_type_bits, handles, CallSignature, Tensor, TypeSignature};
use crate::types::ShapedType;
use core::ffi::c_char;
use core::ptr;
use std::ffi::CString;

fn cstring_ptr(value: &CString) -> *const c_char {
    value.as_ptr()
}

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn optional_name_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn static_type(output_type: &ShapedType) -> Result<TypeSignature> {
    let shape = output_type
        .shape()
        .filter(|shape| shape.iter().all(|extent| *extent >= 0))
        .ok_or(Error::InvalidArgument(
            "call output types need static shapes",
        ))?;
    let data_type = output_type.data_type();
    if data_type_bits(data_type).is_none() {
        return Err(Error::UnsupportedDataType(data_type));
    }
    Ok(TypeSignature {
        shape: Some(shape),
        data_type,
    })
}

impl crate::graph::Graph {
/// Calls the `MPSGraph` framework counterpart for `call`.
    pub fn call(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Result<Vec<Tensor>> {
        self.check(input_tensors)?;
        if output_types.is_empty() {
            return Err(Error::InvalidArgument("a call needs at least one output type"));
        }
        let outputs = output_types
            .iter()
            .map(|output_type| static_type(output_type))
            .collect::<Result<Vec<_>>>()?;
        let symbol = CString::new(symbol_name)
            .map_err(|_| Error::OperationFailed("call symbol name contained NUL"))?;
        let name = optional_cstring(name);
        let input_handles = input_tensors
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        let output_type_handles = output_types
            .iter()
            .map(|output_type| output_type.as_ptr())
            .collect::<Vec<_>>();
        let input_ptr = if input_handles.is_empty() {
            ptr::null()
        } else {
            input_handles.as_ptr()
        };
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_call_symbol(
                self.as_ptr(),
                cstring_ptr(&symbol),
                input_ptr,
                input_handles.len(),
                output_type_handles.as_ptr(),
                output_type_handles.len(),
                optional_name_ptr(&name),
            )
        };
        let count = outputs.len();
        self.record_call(CallSignature {
            symbol: symbol_name.to_owned(),
            inputs: input_tensors
                .iter()
                .map(|tensor| TypeSignature::of(tensor))
                .collect(),
            outputs,
        });
        self.outputs(
            box_handle,
            &handles(input_tensors),
            Some(count),
            "failed to create call op",
        )
    }
}
