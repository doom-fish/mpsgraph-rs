use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::Tensor;
use crate::types::collect_owned_tensors;
use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CString;

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn optional_tensor_ptr(tensor: Option<&Tensor>) -> *mut c_void {
    tensor.map_or(ptr::null_mut(), Tensor::as_ptr)
}

fn wrap_tensor_array(box_handle: *mut c_void) -> Option<Vec<Tensor>> {
    if box_handle.is_null() {
        None
    } else {
        Some(collect_owned_tensors(box_handle))
    }
}

/// `MPSGraphRNNActivation` constants.
pub mod rnn_activation {
    pub const NONE: usize = 0;
    pub const RELU: usize = 1;
    pub const TANH: usize = 2;
    pub const SIGMOID: usize = 3;
    pub const HARD_SIGMOID: usize = 4;
}

macro_rules! descriptor_handle {
    ($name:ident) => {
        pub struct $name {
            ptr: *mut c_void,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
                    unsafe { ffi::mpsgraph_object_release(self.ptr) };
                    self.ptr = ptr::null_mut();
                }
            }
        }

        impl $name {
            #[must_use]
            pub(crate) const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

macro_rules! bool_getter_setter {
    ($getter:ident, $setter:ident, $ffi_get:ident, $ffi_set:ident, $msg:literal) => {
        #[must_use]
        pub fn $getter(&self) -> bool {
            // SAFETY: `self.ptr` is a live descriptor handle.
            unsafe { ffi::$ffi_get(self.ptr) }
        }

        pub fn $setter(&self, value: bool) -> Result<()> {
            // SAFETY: `self.ptr` is a live descriptor handle.
            let ok = unsafe { ffi::$ffi_set(self.ptr, value) };
            if ok {
                Ok(())
            } else {
                Err(Error::OperationFailed($msg))
            }
        }
    };
}

macro_rules! activation_getter_setter {
    ($getter:ident, $setter:ident, $ffi_get:ident, $ffi_set:ident, $msg:literal) => {
        #[must_use]
        pub fn $getter(&self) -> usize {
            // SAFETY: `self.ptr` is a live descriptor handle.
            unsafe { ffi::$ffi_get(self.ptr) }
        }

        pub fn $setter(&self, value: usize) -> Result<()> {
            // SAFETY: `self.ptr` is a live descriptor handle.
            let ok = unsafe { ffi::$ffi_set(self.ptr, value) };
            if ok {
                Ok(())
            } else {
                Err(Error::OperationFailed($msg))
            }
        }
    };
}

descriptor_handle!(SingleGateRNNDescriptor);
impl SingleGateRNNDescriptor {
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_single_gate_rnn_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    bool_getter_setter!(
        reverse,
        set_reverse,
        mpsgraph_single_gate_rnn_descriptor_reverse,
        mpsgraph_single_gate_rnn_descriptor_set_reverse,
        "failed to set single-gate RNN reverse"
    );
    bool_getter_setter!(
        bidirectional,
        set_bidirectional,
        mpsgraph_single_gate_rnn_descriptor_bidirectional,
        mpsgraph_single_gate_rnn_descriptor_set_bidirectional,
        "failed to set single-gate RNN bidirectional"
    );
    bool_getter_setter!(
        training,
        set_training,
        mpsgraph_single_gate_rnn_descriptor_training,
        mpsgraph_single_gate_rnn_descriptor_set_training,
        "failed to set single-gate RNN training"
    );
    activation_getter_setter!(
        activation,
        set_activation,
        mpsgraph_single_gate_rnn_descriptor_activation,
        mpsgraph_single_gate_rnn_descriptor_set_activation,
        "failed to set single-gate RNN activation"
    );
}

descriptor_handle!(LSTMDescriptor);
impl LSTMDescriptor {
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_lstm_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    bool_getter_setter!(
        reverse,
        set_reverse,
        mpsgraph_lstm_descriptor_reverse,
        mpsgraph_lstm_descriptor_set_reverse,
        "failed to set LSTM reverse"
    );
    bool_getter_setter!(
        bidirectional,
        set_bidirectional,
        mpsgraph_lstm_descriptor_bidirectional,
        mpsgraph_lstm_descriptor_set_bidirectional,
        "failed to set LSTM bidirectional"
    );
    bool_getter_setter!(
        produce_cell,
        set_produce_cell,
        mpsgraph_lstm_descriptor_produce_cell,
        mpsgraph_lstm_descriptor_set_produce_cell,
        "failed to set LSTM produceCell"
    );
    bool_getter_setter!(
        training,
        set_training,
        mpsgraph_lstm_descriptor_training,
        mpsgraph_lstm_descriptor_set_training,
        "failed to set LSTM training"
    );
    bool_getter_setter!(
        forget_gate_last,
        set_forget_gate_last,
        mpsgraph_lstm_descriptor_forget_gate_last,
        mpsgraph_lstm_descriptor_set_forget_gate_last,
        "failed to set LSTM forgetGateLast"
    );
    activation_getter_setter!(
        input_gate_activation,
        set_input_gate_activation,
        mpsgraph_lstm_descriptor_input_gate_activation,
        mpsgraph_lstm_descriptor_set_input_gate_activation,
        "failed to set LSTM inputGateActivation"
    );
    activation_getter_setter!(
        forget_gate_activation,
        set_forget_gate_activation,
        mpsgraph_lstm_descriptor_forget_gate_activation,
        mpsgraph_lstm_descriptor_set_forget_gate_activation,
        "failed to set LSTM forgetGateActivation"
    );
    activation_getter_setter!(
        cell_gate_activation,
        set_cell_gate_activation,
        mpsgraph_lstm_descriptor_cell_gate_activation,
        mpsgraph_lstm_descriptor_set_cell_gate_activation,
        "failed to set LSTM cellGateActivation"
    );
    activation_getter_setter!(
        output_gate_activation,
        set_output_gate_activation,
        mpsgraph_lstm_descriptor_output_gate_activation,
        mpsgraph_lstm_descriptor_set_output_gate_activation,
        "failed to set LSTM outputGateActivation"
    );
    activation_getter_setter!(
        activation,
        set_activation,
        mpsgraph_lstm_descriptor_activation,
        mpsgraph_lstm_descriptor_set_activation,
        "failed to set LSTM activation"
    );
}

descriptor_handle!(GRUDescriptor);
impl GRUDescriptor {
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_gru_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    bool_getter_setter!(
        reverse,
        set_reverse,
        mpsgraph_gru_descriptor_reverse,
        mpsgraph_gru_descriptor_set_reverse,
        "failed to set GRU reverse"
    );
    bool_getter_setter!(
        bidirectional,
        set_bidirectional,
        mpsgraph_gru_descriptor_bidirectional,
        mpsgraph_gru_descriptor_set_bidirectional,
        "failed to set GRU bidirectional"
    );
    bool_getter_setter!(
        training,
        set_training,
        mpsgraph_gru_descriptor_training,
        mpsgraph_gru_descriptor_set_training,
        "failed to set GRU training"
    );
    bool_getter_setter!(
        reset_gate_first,
        set_reset_gate_first,
        mpsgraph_gru_descriptor_reset_gate_first,
        mpsgraph_gru_descriptor_set_reset_gate_first,
        "failed to set GRU resetGateFirst"
    );
    bool_getter_setter!(
        reset_after,
        set_reset_after,
        mpsgraph_gru_descriptor_reset_after,
        mpsgraph_gru_descriptor_set_reset_after,
        "failed to set GRU resetAfter"
    );
    bool_getter_setter!(
        flip_z,
        set_flip_z,
        mpsgraph_gru_descriptor_flip_z,
        mpsgraph_gru_descriptor_set_flip_z,
        "failed to set GRU flipZ"
    );
    activation_getter_setter!(
        update_gate_activation,
        set_update_gate_activation,
        mpsgraph_gru_descriptor_update_gate_activation,
        mpsgraph_gru_descriptor_set_update_gate_activation,
        "failed to set GRU updateGateActivation"
    );
    activation_getter_setter!(
        reset_gate_activation,
        set_reset_gate_activation,
        mpsgraph_gru_descriptor_reset_gate_activation,
        mpsgraph_gru_descriptor_set_reset_gate_activation,
        "failed to set GRU resetGateActivation"
    );
    activation_getter_setter!(
        output_gate_activation,
        set_output_gate_activation,
        mpsgraph_gru_descriptor_output_gate_activation,
        mpsgraph_gru_descriptor_set_output_gate_activation,
        "failed to set GRU outputGateActivation"
    );
}

impl crate::graph::Graph {
    #[allow(clippy::too_many_arguments)]
    pub fn single_gate_rnn(
        &self,
        source: &Tensor,
        recurrent_weight: &Tensor,
        input_weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        init_state: Option<&Tensor>,
        mask: Option<&Tensor>,
        descriptor: &SingleGateRNNDescriptor,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_single_gate_rnn(
                self.as_ptr(),
                source.as_ptr(),
                recurrent_weight.as_ptr(),
                optional_tensor_ptr(input_weight),
                optional_tensor_ptr(bias),
                optional_tensor_ptr(init_state),
                optional_tensor_ptr(mask),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn lstm(
        &self,
        source: &Tensor,
        recurrent_weight: &Tensor,
        input_weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        init_state: Option<&Tensor>,
        init_cell: Option<&Tensor>,
        mask: Option<&Tensor>,
        peephole: Option<&Tensor>,
        descriptor: &LSTMDescriptor,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_lstm(
                self.as_ptr(),
                source.as_ptr(),
                recurrent_weight.as_ptr(),
                optional_tensor_ptr(input_weight),
                optional_tensor_ptr(bias),
                optional_tensor_ptr(init_state),
                optional_tensor_ptr(init_cell),
                optional_tensor_ptr(mask),
                optional_tensor_ptr(peephole),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gru(
        &self,
        source: &Tensor,
        recurrent_weight: &Tensor,
        input_weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        init_state: Option<&Tensor>,
        mask: Option<&Tensor>,
        secondary_bias: Option<&Tensor>,
        descriptor: &GRUDescriptor,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_gru(
                self.as_ptr(),
                source.as_ptr(),
                recurrent_weight.as_ptr(),
                optional_tensor_ptr(input_weight),
                optional_tensor_ptr(bias),
                optional_tensor_ptr(init_state),
                optional_tensor_ptr(mask),
                optional_tensor_ptr(secondary_bias),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }
}
