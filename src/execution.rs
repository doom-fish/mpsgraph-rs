use crate::data::TensorData;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{Executable, FeedDescription, Graph, Tensor};
use crate::types::{
    collect_owned_tensors, collect_shaped_type_array_box, collect_tensor_data_array_box, ShapedType,
};
use apple_metal::{CommandQueue, MetalDevice};
use core::ffi::c_void;
use core::ptr;
use std::ffi::CString;

fn release_handle(ptr: &mut *mut c_void) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
        unsafe { ffi::mpsgraph_object_release(*ptr) };
        *ptr = ptr::null_mut();
    }
}

fn copy_string(
    len: unsafe extern "C" fn(*mut c_void) -> usize,
    copy: unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> bool,
    handle: *mut c_void,
) -> Result<String> {
    // SAFETY: the function pointers belong to Swift shims that treat `handle` as immutable for the duration of the call.
    let len = unsafe { len(handle) };
    let mut bytes = vec![0_u8; len];
    // SAFETY: the buffer is valid for exactly `len` bytes.
    let ok = unsafe { copy(handle, bytes.as_mut_ptr(), len) };
    if ok {
        String::from_utf8(bytes)
            .map_err(|_| Error::OperationFailed("bridge returned invalid UTF-8"))
    } else {
        Err(Error::OperationFailed("failed to copy string from bridge"))
    }
}

/// `MPSGraphOptions` constants.
pub mod graph_options {
/// Mirrors the `MPSGraph` framework constant `NONE`.
    pub const NONE: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `SYNCHRONIZE_RESULTS`.
    pub const SYNCHRONIZE_RESULTS: u64 = 1;
/// Mirrors the `MPSGraph` framework constant `VERBOSE`.
    pub const VERBOSE: u64 = 2;
/// Mirrors the `MPSGraph` framework constant `DEFAULT`.
    pub const DEFAULT: u64 = SYNCHRONIZE_RESULTS;
}

/// `MPSGraphOptimization` constants.
pub mod optimization {
/// Mirrors the `MPSGraph` framework constant `LEVEL0`.
    pub const LEVEL0: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `LEVEL1`.
    pub const LEVEL1: u64 = 1;
}

/// `MPSGraphOptimizationProfile` constants.
pub mod optimization_profile {
/// Mirrors the `MPSGraph` framework constant `PERFORMANCE`.
    pub const PERFORMANCE: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `POWER_EFFICIENCY`.
    pub const POWER_EFFICIENCY: u64 = 1;
}

/// `MPSGraphReducedPrecisionFastMath` bit flags.
pub mod reduced_precision_fast_math {
/// Mirrors the `MPSGraph` framework constant `NONE`.
    pub const NONE: usize = 0;
/// Mirrors the `MPSGraph` framework constant `ALLOW_FP16_CONV2D_WINOGRAD_TRANSFORM_INTERMEDIATE`.
    pub const ALLOW_FP16_CONV2D_WINOGRAD_TRANSFORM_INTERMEDIATE: usize = 1 << 1;
/// Mirrors the `MPSGraph` framework constant `ALLOW_FP16_INTERMEDIATES`.
    pub const ALLOW_FP16_INTERMEDIATES: usize = ALLOW_FP16_CONV2D_WINOGRAD_TRANSFORM_INTERMEDIATE;
/// Mirrors the `MPSGraph` framework constant `DEFAULT`.
    pub const DEFAULT: usize = NONE;
}

/// `MPSGraphDeploymentPlatform` constants.
pub mod deployment_platform {
/// Mirrors the `MPSGraph` framework constant `MACOS`.
    pub const MACOS: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `IOS`.
    pub const IOS: u64 = 1;
/// Mirrors the `MPSGraph` framework constant `TVOS`.
    pub const TVOS: u64 = 2;
/// Mirrors the `MPSGraph` framework constant `VISIONOS`.
    pub const VISIONOS: u64 = 3;
}

/// Safe owner for `MPSGraphCompilationDescriptor`.
pub struct CompilationDescriptor {
    ptr: *mut c_void,
}

unsafe impl Send for CompilationDescriptor {}
unsafe impl Sync for CompilationDescriptor {}

impl Drop for CompilationDescriptor {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl CompilationDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_compilation_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Calls the `MPSGraph` framework counterpart for `disable_type_inference`.
    pub fn disable_type_inference(&self) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_compilation_descriptor_disable_type_inference(self.ptr) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to disable type inference"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `optimization_level`.
    #[must_use]
    pub fn optimization_level(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_compilation_descriptor_optimization_level(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_optimization_level`.
    pub fn set_optimization_level(&self, value: u64) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok =
            unsafe { ffi::mpsgraph_compilation_descriptor_set_optimization_level(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set optimization level"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `wait_for_compilation_completion`.
    #[must_use]
    pub fn wait_for_compilation_completion(&self) -> bool {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_compilation_descriptor_wait_for_completion(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_wait_for_compilation_completion`.
    pub fn set_wait_for_compilation_completion(&self, value: bool) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe {
            ffi::mpsgraph_compilation_descriptor_set_wait_for_completion(self.ptr, value)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set waitForCompilationCompletion",
            ))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `optimization_profile`.
    #[must_use]
    pub fn optimization_profile(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_compilation_descriptor_optimization_profile(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_optimization_profile`.
    pub fn set_optimization_profile(&self, value: u64) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe {
            ffi::mpsgraph_compilation_descriptor_set_optimization_profile(self.ptr, value)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set optimization profile"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `reduced_precision_fast_math`.
    #[must_use]
    pub fn reduced_precision_fast_math(&self) -> usize {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_compilation_descriptor_reduced_precision_fast_math(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_reduced_precision_fast_math`.
    pub fn set_reduced_precision_fast_math(&self, value: usize) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe {
            ffi::mpsgraph_compilation_descriptor_set_reduced_precision_fast_math(self.ptr, value)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set reducedPrecisionFastMath",
            ))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `set_callable`.
    pub fn set_callable(&self, symbol_name: &str, executable: Option<&Executable>) -> Result<()> {
        let symbol_name = CString::new(symbol_name)
            .map_err(|_| Error::OperationFailed("call symbol name contained NUL"))?;
        let executable_ptr = executable.map_or(ptr::null_mut(), Executable::as_ptr);
        // SAFETY: all handles remain valid for the duration of the call.
        let ok = unsafe {
            ffi::mpsgraph_compilation_descriptor_set_callable(
                self.ptr,
                symbol_name.as_ptr(),
                executable_ptr,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set compilation descriptor callable",
            ))
        }
    }
}

/// Safe owner for `MPSGraphExecutionDescriptor`.
pub struct ExecutionDescriptor {
    ptr: *mut c_void,
}

unsafe impl Send for ExecutionDescriptor {}
unsafe impl Sync for ExecutionDescriptor {}

impl Drop for ExecutionDescriptor {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl ExecutionDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_execution_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Calls the `MPSGraph` framework counterpart for `wait_until_completed`.
    #[must_use]
    pub fn wait_until_completed(&self) -> bool {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_execution_descriptor_wait_until_completed(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_wait_until_completed`.
    pub fn set_wait_until_completed(&self, value: bool) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok =
            unsafe { ffi::mpsgraph_execution_descriptor_set_wait_until_completed(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set waitUntilCompleted"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `compilation_descriptor`.
    #[must_use]
    pub fn compilation_descriptor(&self) -> Option<CompilationDescriptor> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ptr = unsafe { ffi::mpsgraph_execution_descriptor_compilation_descriptor(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(CompilationDescriptor { ptr })
        }
    }

/// Calls the `MPSGraph` framework counterpart for `set_compilation_descriptor`.
    pub fn set_compilation_descriptor(
        &self,
        descriptor: Option<&CompilationDescriptor>,
    ) -> Result<()> {
        let descriptor_ptr = descriptor.map_or(ptr::null_mut(), CompilationDescriptor::as_ptr);
        // SAFETY: all handles remain valid for the duration of the call.
        let ok = unsafe {
            ffi::mpsgraph_execution_descriptor_set_compilation_descriptor(self.ptr, descriptor_ptr)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set compilation descriptor",
            ))
        }
    }
}

/// Safe owner for `MPSGraphExecutableExecutionDescriptor`.
pub struct ExecutableExecutionDescriptor {
    ptr: *mut c_void,
}

unsafe impl Send for ExecutableExecutionDescriptor {}
unsafe impl Sync for ExecutableExecutionDescriptor {}

impl Drop for ExecutableExecutionDescriptor {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl ExecutableExecutionDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_executable_execution_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Calls the `MPSGraph` framework counterpart for `wait_until_completed`.
    #[must_use]
    pub fn wait_until_completed(&self) -> bool {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_executable_execution_descriptor_wait_until_completed(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_wait_until_completed`.
    pub fn set_wait_until_completed(&self, value: bool) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe {
            ffi::mpsgraph_executable_execution_descriptor_set_wait_until_completed(self.ptr, value)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set executable waitUntilCompleted",
            ))
        }
    }
}

/// Safe owner for `MPSGraphExecutableSerializationDescriptor`.
pub struct ExecutableSerializationDescriptor {
    ptr: *mut c_void,
}

unsafe impl Send for ExecutableSerializationDescriptor {}
unsafe impl Sync for ExecutableSerializationDescriptor {}

impl Drop for ExecutableSerializationDescriptor {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl ExecutableSerializationDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: pure constructor.
        let ptr = unsafe { ffi::mpsgraph_executable_serialization_descriptor_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Calls the `MPSGraph` framework counterpart for `append`.
    #[must_use]
    pub fn append(&self) -> bool {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_executable_serialization_descriptor_append(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_append`.
    pub fn set_append(&self, value: bool) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe {
            ffi::mpsgraph_executable_serialization_descriptor_set_append(self.ptr, value)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set append"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `deployment_platform`.
    #[must_use]
    pub fn deployment_platform(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_executable_serialization_descriptor_deployment_platform(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_deployment_platform`.
    pub fn set_deployment_platform(&self, value: u64) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe {
            ffi::mpsgraph_executable_serialization_descriptor_set_deployment_platform(
                self.ptr, value,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set deployment platform"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `minimum_deployment_target`.
    pub fn minimum_deployment_target(&self) -> Result<String> {
        copy_string(
            ffi::mpsgraph_executable_serialization_descriptor_minimum_deployment_target_len,
            ffi::mpsgraph_executable_serialization_descriptor_copy_minimum_deployment_target,
            self.ptr,
        )
    }

/// Calls the `MPSGraph` framework counterpart for `set_minimum_deployment_target`.
    pub fn set_minimum_deployment_target(&self, value: &str) -> Result<()> {
        let value = CString::new(value)
            .map_err(|_| Error::OperationFailed("minimum deployment target contained NUL"))?;
        // SAFETY: the CString stays alive for the duration of the call.
        let ok = unsafe {
            ffi::mpsgraph_executable_serialization_descriptor_set_minimum_deployment_target(
                self.ptr,
                value.as_ptr(),
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set minimum deployment target",
            ))
        }
    }
}

impl Graph {
    /// Return the graph's `MPSGraphOptions` bitmask.
    #[must_use]
    pub fn options(&self) -> u64 {
        // SAFETY: `self` owns a live graph handle.
        unsafe { ffi::mpsgraph_graph_options(self.as_ptr()) }
    }

    /// Replace the graph's options bitmask.
    pub fn set_options(&self, options: u64) -> Result<()> {
        // SAFETY: `self` owns a live graph handle.
        let ok = unsafe { ffi::mpsgraph_graph_set_options(self.as_ptr(), options) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set graph options"))
        }
    }

    /// Return the graph's placeholder tensors in insertion order.
    #[must_use]
    pub fn placeholder_tensors(&self) -> Vec<Tensor> {
        // SAFETY: `self` owns a live graph handle.
        let box_handle = unsafe { ffi::mpsgraph_graph_placeholder_tensors(self.as_ptr()) };
        collect_owned_tensors(box_handle)
    }

    /// Compile the graph with an optional compilation descriptor.
    #[must_use]
    pub fn compile_with_descriptor(
        &self,
        device: Option<&MetalDevice>,
        feeds: &[FeedDescription<'_>],
        targets: &[&Tensor],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Option<Executable> {
        let feed_tensors = feeds
            .iter()
            .map(|feed| feed.tensor.as_ptr())
            .collect::<Vec<_>>();
        let shape_lengths = feeds
            .iter()
            .map(|feed| feed.shape.len())
            .collect::<Vec<_>>();
        let data_types = feeds.iter().map(|feed| feed.data_type).collect::<Vec<_>>();
        let flat_shapes = feeds
            .iter()
            .flat_map(|feed| feed.shape.iter().copied())
            .collect::<Vec<_>>();
        let target_tensors = targets
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        let device_ptr = device.map_or(ptr::null_mut(), MetalDevice::as_ptr);
        let descriptor_ptr = descriptor.map_or(ptr::null_mut(), CompilationDescriptor::as_ptr);

        // SAFETY: all pointer arrays stay alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_compile_with_descriptor(
                self.as_ptr(),
                device_ptr,
                feed_tensors.as_ptr(),
                feeds.len(),
                flat_shapes.as_ptr(),
                shape_lengths.as_ptr(),
                data_types.as_ptr(),
                target_tensors.as_ptr(),
                targets.len(),
                descriptor_ptr,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Executable::from_raw(ptr, targets.len()))
        }
    }
}

impl Executable {
    /// Return the executable's `MPSGraphOptions` bitmask.
    #[must_use]
    pub fn options(&self) -> u64 {
        // SAFETY: `self` owns a live executable handle.
        unsafe { ffi::mpsgraph_executable_options(self.as_ptr()) }
    }

    /// Replace the executable's options bitmask.
    pub fn set_options(&self, options: u64) -> Result<()> {
        // SAFETY: `self` owns a live executable handle.
        let ok = unsafe { ffi::mpsgraph_executable_set_options(self.as_ptr(), options) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set executable options"))
        }
    }

    /// Return feed tensors if this executable was compiled from a graph.
    #[must_use]
    pub fn feed_tensors(&self) -> Vec<Tensor> {
        // SAFETY: `self` owns a live executable handle.
        let box_handle = unsafe { ffi::mpsgraph_executable_feed_tensors(self.as_ptr()) };
        collect_owned_tensors(box_handle)
    }

    /// Return target tensors if this executable was compiled from a graph.
    #[must_use]
    pub fn target_tensors(&self) -> Vec<Tensor> {
        // SAFETY: `self` owns a live executable handle.
        let box_handle = unsafe { ffi::mpsgraph_executable_target_tensors(self.as_ptr()) };
        collect_owned_tensors(box_handle)
    }

    /// Specialize the executable for the provided input types.
    pub fn specialize(
        &self,
        device: Option<&MetalDevice>,
        input_types: &[&ShapedType],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Result<()> {
        let input_type_handles = input_types
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let device_ptr = device.map_or(ptr::null_mut(), MetalDevice::as_ptr);
        let descriptor_ptr = descriptor.map_or(ptr::null_mut(), CompilationDescriptor::as_ptr);

        // SAFETY: all pointer arrays stay alive for the duration of the call.
        let ok = unsafe {
            ffi::mpsgraph_executable_specialize(
                self.as_ptr(),
                device_ptr,
                input_type_handles.as_ptr(),
                input_types.len(),
                descriptor_ptr,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to specialize executable"))
        }
    }

    /// Query specialized output types for the provided input types.
    pub fn output_types(
        &self,
        device: Option<&MetalDevice>,
        input_types: &[&ShapedType],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Result<Vec<ShapedType>> {
        let input_type_handles = input_types
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let device_ptr = device.map_or(ptr::null_mut(), MetalDevice::as_ptr);
        let descriptor_ptr = descriptor.map_or(ptr::null_mut(), CompilationDescriptor::as_ptr);

        // SAFETY: all pointer arrays stay alive for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_executable_get_output_types(
                self.as_ptr(),
                device_ptr,
                input_type_handles.as_ptr(),
                input_types.len(),
                descriptor_ptr,
            )
        };
        if box_handle.is_null() {
            Err(Error::OperationFailed(
                "failed to get executable output types",
            ))
        } else {
            Ok(collect_shaped_type_array_box(box_handle))
        }
    }

    /// Run the executable with an optional execution descriptor and optional preallocated results.
    pub fn run_with_descriptor(
        &self,
        command_queue: &CommandQueue,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Result<Vec<TensorData>> {
        let input_handles = inputs
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let result_handles = results
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_ptr())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let descriptor_ptr =
            descriptor.map_or(ptr::null_mut(), ExecutableExecutionDescriptor::as_ptr);

        // SAFETY: all pointer arrays stay alive for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_executable_run_with_descriptor(
                self.as_ptr(),
                command_queue.as_ptr(),
                input_handles.as_ptr(),
                inputs.len(),
                result_handles.as_ptr(),
                result_handles.len(),
                descriptor_ptr,
            )
        };
        if box_handle.is_null() {
            Err(Error::OperationFailed("failed to run executable"))
        } else {
            Ok(collect_tensor_data_array_box(box_handle))
        }
    }

    /// Asynchronously run the executable with an optional execution descriptor.
    pub fn run_async_with_descriptor(
        &self,
        command_queue: &CommandQueue,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        descriptor: Option<&ExecutableExecutionDescriptor>,
    ) -> Result<Vec<TensorData>> {
        let input_handles = inputs
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let result_handles = results
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_ptr())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let descriptor_ptr =
            descriptor.map_or(ptr::null_mut(), ExecutableExecutionDescriptor::as_ptr);

        // SAFETY: all pointer arrays stay alive for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_executable_run_async_with_descriptor(
                self.as_ptr(),
                command_queue.as_ptr(),
                input_handles.as_ptr(),
                inputs.len(),
                result_handles.as_ptr(),
                result_handles.len(),
                descriptor_ptr,
            )
        };
        if box_handle.is_null() {
            Err(Error::OperationFailed(
                "failed to run executable asynchronously",
            ))
        } else {
            Ok(collect_tensor_data_array_box(box_handle))
        }
    }

    /// Serialize the executable to an `.mpsgraphpackage` path.
    pub fn serialize_package(
        &self,
        path: &str,
        descriptor: Option<&ExecutableSerializationDescriptor>,
    ) -> Result<()> {
        let path =
            CString::new(path).map_err(|_| Error::OperationFailed("package path contained NUL"))?;
        let descriptor_ptr =
            descriptor.map_or(ptr::null_mut(), ExecutableSerializationDescriptor::as_ptr);
        // SAFETY: the CString stays alive for the duration of the call.
        let ok = unsafe {
            ffi::mpsgraph_executable_serialize_package(self.as_ptr(), path.as_ptr(), descriptor_ptr)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to serialize executable package",
            ))
        }
    }

    /// Load an executable from an existing `.mpsgraphpackage`.
    pub fn from_package(path: &str, descriptor: Option<&CompilationDescriptor>) -> Result<Self> {
        let path =
            CString::new(path).map_err(|_| Error::OperationFailed("package path contained NUL"))?;
        let descriptor_ptr = descriptor.map_or(ptr::null_mut(), CompilationDescriptor::as_ptr);
        // SAFETY: the CString stays alive for the duration of the call.
        let ptr =
            unsafe { ffi::mpsgraph_executable_new_with_package(path.as_ptr(), descriptor_ptr) };
        if ptr.is_null() {
            return Err(Error::OperationFailed("failed to load executable package"));
        }
        let output_count = {
            // SAFETY: `ptr` is a live executable handle returned just above.
            let box_handle = unsafe { ffi::mpsgraph_executable_target_tensors(ptr) };
            collect_owned_tensors(box_handle).len()
        };
        Ok(Self::from_raw(ptr, output_count))
    }
}
