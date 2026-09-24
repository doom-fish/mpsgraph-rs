use crate::checks;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{handles, Tensor};
use core::cell::Cell;
use core::ffi::{c_char, c_void};
use core::marker::PhantomData;
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


/// `MPSGraphRNNActivation` constants.
pub mod rnn_activation {
/// Mirrors the `MPSGraph` framework constant `NONE`.
    pub const NONE: usize = 0;
/// Mirrors the `MPSGraph` framework constant `RELU`.
    pub const RELU: usize = 1;
/// Mirrors the `MPSGraph` framework constant `TANH`.
    pub const TANH: usize = 2;
/// Mirrors the `MPSGraph` framework constant `SIGMOID`.
    pub const SIGMOID: usize = 3;
/// Mirrors the `MPSGraph` framework constant `HARD_SIGMOID`.
    pub const HARD_SIGMOID: usize = 4;
}

macro_rules! descriptor_handle {
    ($name:ident) => {
/// Mirrors the `MPSGraph` framework counterpart for this type.
        pub struct $name {
            ptr: *mut c_void,
            _not_sync: PhantomData<Cell<()>>,
        }

        unsafe impl Send for $name {}

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
/// Calls the `MPSGraph` framework counterpart for this method.
        #[must_use]
        pub fn $getter(&self) -> bool {
            // SAFETY: `self.ptr` is a live descriptor handle.
            unsafe { ffi::$ffi_get(self.ptr) }
        }

/// Calls the `MPSGraph` framework counterpart for this method.
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
/// Calls the `MPSGraph` framework counterpart for this method.
        #[must_use]
        pub fn $getter(&self) -> usize {
            // SAFETY: `self.ptr` is a live descriptor handle.
            unsafe { ffi::$ffi_get(self.ptr) }
        }

/// Calls the `MPSGraph` framework counterpart for this method.
        pub fn $setter(&self, value: usize) -> Result<()> {
            if value > rnn_activation::HARD_SIGMOID {
                return Err(Error::InvalidArgument("unknown MPSGraphRNNActivation"));
            }
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
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_single_gate_rnn_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _not_sync: PhantomData,
            })
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
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_lstm_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _not_sync: PhantomData,
            })
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
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_gru_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _not_sync: PhantomData,
            })
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

fn scaled(factor: isize, extent: isize) -> isize {
    if extent < 0 {
        -1
    } else {
        factor.checked_mul(extent).unwrap_or(-1)
    }
}

struct RecurrentInputs<'a> {
    gates: isize,
    bidirectional: bool,
    source: &'a Tensor,
    recurrent_weight: &'a Tensor,
    input_weight: Option<&'a Tensor>,
    bias: Option<&'a Tensor>,
    init_state: Option<&'a Tensor>,
    mask: Option<&'a Tensor>,
}

struct RecurrentShape {
    batch: isize,
    hidden: isize,
    directions: isize,
}

impl<'a> RecurrentInputs<'a> {
    fn check(&self, extra: &[Option<&Tensor>]) -> Result<RecurrentShape> {
        checks::float_tensor(self.source, "recurrent layers need a floating-point source")?;
        let optional = [self.input_weight, self.bias, self.init_state, self.mask];
        for tensor in core::iter::once(self.recurrent_weight)
            .chain(optional.into_iter().flatten())
            .chain(extra.iter().copied().flatten())
        {
            checks::same_data_type(
                self.source,
                tensor,
                "every recurrent-layer input needs the source's data type",
            )?;
        }
        let source = checks::rank_exactly(
            self.source,
            3,
            "recurrent-layer sources must have layout [T, N, I]",
        )?;
        let (steps, batch, inputs) = source.map_or((-1, -1, -1), |dims| (dims[0], dims[1], dims[2]));
        let directions: isize = if self.bidirectional { 2 } else { 1 };
        let recurrent = checks::rank_exactly(
            self.recurrent_weight,
            if self.bidirectional { 3 } else { 2 },
            "the recurrent weight must have layout [G*H, H], or [2, G*H, H] when bidirectional",
        )?;
        let hidden = recurrent.map_or(-1, |dims| dims[dims.len() - 1]);
        let gated = scaled(self.gates, hidden);
        let expected_recurrent: &[isize] = if self.bidirectional {
            &[2, gated, hidden]
        } else {
            &[gated, hidden]
        };
        checks::dims_match(
            self.recurrent_weight,
            expected_recurrent,
            "the recurrent weight must have layout [G*H, H], or [2, G*H, H] when bidirectional",
        )?;
        let all_gates = scaled(directions, gated);
        match self.input_weight {
            Some(weight) => checks::dims_match(
                weight,
                &[all_gates, inputs],
                "the input weight must have layout [G*H, I], or [2*G*H, I] when bidirectional",
            )?,
            None => checks::dims_match(
                self.source,
                &[steps, batch, all_gates],
                "without an input weight the source must have layout [T, N, G*H], or [T, N, 2*G*H] when bidirectional",
            )?,
        }
        if let Some(bias) = self.bias {
            checks::dims_match(
                bias,
                &[all_gates],
                "the bias must have layout [G*H], or [2*G*H] when bidirectional",
            )?;
        }
        let states = scaled(directions, hidden);
        if let Some(state) = self.init_state {
            checks::dims_match(
                state,
                &[batch, states],
                "the initial state must have layout [N, H], or [N, 2H] when bidirectional",
            )?;
        }
        if let Some(mask_dims) = self.mask.and_then(Tensor::shape) {
            if mask_dims.len() != 3 || !checks::broadcastable(&mask_dims, &[steps, batch, states]) {
                return Err(Error::InvalidShape(
                    "the mask must broadcast to [T, N, H], or [T, N, 2H] when bidirectional",
                ));
            }
        }
        Ok(RecurrentShape {
            batch,
            hidden,
            directions,
        })
    }

    fn tensors(&self, extra: &[Option<&'a Tensor>]) -> Vec<&'a Tensor> {
        [self.source, self.recurrent_weight]
            .into_iter()
            .chain(
                [self.input_weight, self.bias, self.init_state, self.mask]
                    .into_iter()
                    .flatten(),
            )
            .chain(extra.iter().copied().flatten())
            .collect()
    }
}

impl crate::graph::Graph {
/// Calls the `MPSGraph` framework counterpart for `single_gate_rnn`.
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
    ) -> Result<Vec<Tensor>> {
        let inputs = RecurrentInputs {
            gates: 1,
            bidirectional: descriptor.bidirectional(),
            source,
            recurrent_weight,
            input_weight,
            bias,
            init_state,
            mask,
        };
        let tensors = inputs.tensors(&[]);
        self.check(&tensors)?;
        inputs.check(&[])?;
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
        self.outputs(
            box_handle,
            &handles(&tensors),
            None,
            "MPSGraph did not create the RNN",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `lstm`.
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
    ) -> Result<Vec<Tensor>> {
        let inputs = RecurrentInputs {
            gates: 4,
            bidirectional: descriptor.bidirectional(),
            source,
            recurrent_weight,
            input_weight,
            bias,
            init_state,
            mask,
        };
        let extra = [init_cell, peephole];
        let tensors = inputs.tensors(&extra);
        self.check(&tensors)?;
        let shape = inputs.check(&extra)?;
        if let Some(cell) = init_cell {
            checks::dims_match(
                cell,
                &[shape.batch, scaled(shape.directions, shape.hidden)],
                "the initial cell must have layout [N, H], or [N, 2H] when bidirectional",
            )?;
        }
        if let Some(peephole) = peephole {
            if inputs.bidirectional {
                return Err(Error::Unsupported(
                    "MPSGraph rejects every peephole shape for bidirectional LSTMs",
                ));
            }
            if init_cell.is_none() {
                return Err(Error::InvalidArgument(
                    "an LSTM peephole needs an initial cell; MPSGraph aborts without one",
                ));
            }
            checks::dims_match(
                peephole,
                &[scaled(4, shape.hidden)],
                "the peephole must have layout [4H]",
            )?;
        }
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
        self.outputs(
            box_handle,
            &handles(&tensors),
            None,
            "MPSGraph did not create the LSTM",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `gru`.
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
    ) -> Result<Vec<Tensor>> {
        let inputs = RecurrentInputs {
            gates: 3,
            bidirectional: descriptor.bidirectional(),
            source,
            recurrent_weight,
            input_weight,
            bias,
            init_state,
            mask,
        };
        let extra = [secondary_bias];
        let tensors = inputs.tensors(&extra);
        self.check(&tensors)?;
        let shape = inputs.check(&extra)?;
        if let Some(secondary_bias) = secondary_bias {
            if !descriptor.reset_after() {
                return Err(Error::InvalidArgument(
                    "a GRU secondary bias needs reset_after",
                ));
            }
            checks::dims_match(
                secondary_bias,
                &[scaled(shape.directions, shape.hidden)],
                "the secondary bias must have layout [H], or [2H] when bidirectional",
            )?;
        }
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
        self.outputs(
            box_handle,
            &handles(&tensors),
            None,
            "MPSGraph did not create the GRU",
        )
    }
}
