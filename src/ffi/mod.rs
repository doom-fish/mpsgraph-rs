use core::ffi::{c_char, c_void};

unsafe extern "C" {
    pub fn mpsgraph_object_release(handle: *mut c_void);

    pub fn mpsgraph_tensor_data_new_with_bytes(
        device_handle: *mut c_void,
        bytes: *const c_void,
        byte_len: usize,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mpsgraph_tensor_data_new_with_buffer(
        buffer_handle: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mpsgraph_tensor_data_data_type(handle: *mut c_void) -> u32;
    pub fn mpsgraph_tensor_data_shape_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_tensor_data_copy_shape(handle: *mut c_void, out_shape: *mut usize);
    pub fn mpsgraph_tensor_data_read_bytes(
        handle: *mut c_void,
        dst: *mut c_void,
        dst_len: usize,
    ) -> bool;
    pub fn mpsgraph_tensor_data_device(handle: *mut c_void) -> *mut c_void;

    pub fn mpsgraph_graph_new() -> *mut c_void;
    pub fn mpsgraph_graph_placeholder(
        graph_handle: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_constant_data(
        graph_handle: *mut c_void,
        bytes: *const c_void,
        byte_len: usize,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_constant_scalar(
        graph_handle: *mut c_void,
        scalar: f64,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_constant_scalar_shaped(
        graph_handle: *mut c_void,
        scalar: f64,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut c_void;

    pub fn mpsgraph_graph_addition(
        graph_handle: *mut c_void,
        primary_tensor: *mut c_void,
        secondary_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_subtraction(
        graph_handle: *mut c_void,
        primary_tensor: *mut c_void,
        secondary_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_multiplication(
        graph_handle: *mut c_void,
        primary_tensor: *mut c_void,
        secondary_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_division(
        graph_handle: *mut c_void,
        primary_tensor: *mut c_void,
        secondary_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_matrix_multiplication(
        graph_handle: *mut c_void,
        primary_tensor: *mut c_void,
        secondary_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_relu(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_sigmoid(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_softmax(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_reshape(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_transpose(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        permutation: *const usize,
        permutation_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_slice(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        dimension: usize,
        start: isize,
        length: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_broadcast(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_reduction_sum(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axes: *const usize,
        axes_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_reduction_maximum(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axes: *const usize,
        axes_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_reduction_minimum(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axes: *const usize,
        axes_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_mean(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axes: *const usize,
        axes_len: usize,
        name: *const c_char,
    ) -> *mut c_void;

    pub fn mpsgraph_convolution2d_descriptor_new(
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        groups: usize,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
        padding_style: usize,
        data_layout: usize,
        weights_layout: usize,
    ) -> *mut c_void;
    pub fn mpsgraph_pooling2d_descriptor_new(
        kernel_width: usize,
        kernel_height: usize,
        stride_in_x: usize,
        stride_in_y: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
        padding_style: usize,
        data_layout: usize,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_convolution2d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        weights_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_max_pooling2d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_normalize(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        mean_tensor: *mut c_void,
        variance_tensor: *mut c_void,
        gamma_tensor: *mut c_void,
        beta_tensor: *mut c_void,
        epsilon: f32,
        name: *const c_char,
    ) -> *mut c_void;

    pub fn mpsgraph_graph_run(
        graph_handle: *mut c_void,
        feed_tensors: *const *mut c_void,
        feed_data: *const *mut c_void,
        feed_count: usize,
        target_tensors: *const *mut c_void,
        target_count: usize,
        out_results: *mut *mut c_void,
    ) -> bool;
    pub fn mpsgraph_graph_run_with_command_queue(
        graph_handle: *mut c_void,
        command_queue_handle: *mut c_void,
        feed_tensors: *const *mut c_void,
        feed_data: *const *mut c_void,
        feed_count: usize,
        target_tensors: *const *mut c_void,
        target_count: usize,
        out_results: *mut *mut c_void,
    ) -> bool;
    pub fn mpsgraph_graph_compile(
        graph_handle: *mut c_void,
        device_handle: *mut c_void,
        feed_tensors: *const *mut c_void,
        feed_count: usize,
        flat_shapes: *const usize,
        shape_lengths: *const usize,
        data_types: *const u32,
        target_tensors: *const *mut c_void,
        target_count: usize,
    ) -> *mut c_void;
    pub fn mpsgraph_executable_run(
        executable_handle: *mut c_void,
        command_queue_handle: *mut c_void,
        input_data: *const *mut c_void,
        input_count: usize,
        output_count: usize,
        out_results: *mut *mut c_void,
    ) -> bool;

    pub fn mpsgraph_tensor_array_box_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_tensor_array_box_get(handle: *mut c_void, index: usize) -> *mut c_void;
    pub fn mpsgraph_tensor_data_array_box_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_tensor_data_array_box_get(handle: *mut c_void, index: usize) -> *mut c_void;
    pub fn mpsgraph_shaped_type_array_box_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_shaped_type_array_box_get(handle: *mut c_void, index: usize) -> *mut c_void;

    pub fn mpsgraph_device_new_with_metal_device(metal_device_handle: *mut c_void) -> *mut c_void;
    pub fn mpsgraph_device_type(handle: *mut c_void) -> u32;

    pub fn mpsgraph_shaped_type_new(
        shape: *const isize,
        shape_len: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mpsgraph_shaped_type_has_shape(handle: *mut c_void) -> bool;
    pub fn mpsgraph_shaped_type_shape_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_shaped_type_copy_shape(handle: *mut c_void, out_shape: *mut isize);
    pub fn mpsgraph_shaped_type_data_type(handle: *mut c_void) -> u32;
    pub fn mpsgraph_shaped_type_set_shape(
        handle: *mut c_void,
        shape: *const isize,
        shape_len: usize,
    ) -> bool;
    pub fn mpsgraph_shaped_type_set_data_type(handle: *mut c_void, data_type: u32) -> bool;
    pub fn mpsgraph_shaped_type_is_equal(handle: *mut c_void, other_handle: *mut c_void) -> bool;

    pub fn mpsgraph_tensor_has_shape(handle: *mut c_void) -> bool;
    pub fn mpsgraph_tensor_shape_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_tensor_copy_shape(handle: *mut c_void, out_shape: *mut isize);
    pub fn mpsgraph_tensor_data_type(handle: *mut c_void) -> u32;
    pub fn mpsgraph_tensor_operation(handle: *mut c_void) -> *mut c_void;

    pub fn mpsgraph_graph_options(handle: *mut c_void) -> u64;
    pub fn mpsgraph_graph_set_options(handle: *mut c_void, raw_value: u64) -> bool;
    pub fn mpsgraph_graph_placeholder_tensors(handle: *mut c_void) -> *mut c_void;

    pub fn mpsgraph_compilation_descriptor_new() -> *mut c_void;
    pub fn mpsgraph_compilation_descriptor_disable_type_inference(handle: *mut c_void) -> bool;
    pub fn mpsgraph_compilation_descriptor_optimization_level(handle: *mut c_void) -> u64;
    pub fn mpsgraph_compilation_descriptor_set_optimization_level(
        handle: *mut c_void,
        raw_value: u64,
    ) -> bool;
    pub fn mpsgraph_compilation_descriptor_wait_for_completion(handle: *mut c_void) -> bool;
    pub fn mpsgraph_compilation_descriptor_set_wait_for_completion(
        handle: *mut c_void,
        value: bool,
    ) -> bool;
    pub fn mpsgraph_compilation_descriptor_optimization_profile(handle: *mut c_void) -> u64;
    pub fn mpsgraph_compilation_descriptor_set_optimization_profile(
        handle: *mut c_void,
        raw_value: u64,
    ) -> bool;
    pub fn mpsgraph_compilation_descriptor_reduced_precision_fast_math(handle: *mut c_void)
    -> usize;
    pub fn mpsgraph_compilation_descriptor_set_reduced_precision_fast_math(
        handle: *mut c_void,
        raw_value: usize,
    ) -> bool;

    pub fn mpsgraph_execution_descriptor_new() -> *mut c_void;
    pub fn mpsgraph_execution_descriptor_wait_until_completed(handle: *mut c_void) -> bool;
    pub fn mpsgraph_execution_descriptor_set_wait_until_completed(
        handle: *mut c_void,
        value: bool,
    ) -> bool;
    pub fn mpsgraph_execution_descriptor_compilation_descriptor(handle: *mut c_void) -> *mut c_void;
    pub fn mpsgraph_execution_descriptor_set_compilation_descriptor(
        handle: *mut c_void,
        compilation_descriptor_handle: *mut c_void,
    ) -> bool;

    pub fn mpsgraph_executable_execution_descriptor_new() -> *mut c_void;
    pub fn mpsgraph_executable_execution_descriptor_wait_until_completed(
        handle: *mut c_void,
    ) -> bool;
    pub fn mpsgraph_executable_execution_descriptor_set_wait_until_completed(
        handle: *mut c_void,
        value: bool,
    ) -> bool;

    pub fn mpsgraph_executable_serialization_descriptor_new() -> *mut c_void;
    pub fn mpsgraph_executable_serialization_descriptor_append(handle: *mut c_void) -> bool;
    pub fn mpsgraph_executable_serialization_descriptor_set_append(
        handle: *mut c_void,
        value: bool,
    ) -> bool;
    pub fn mpsgraph_executable_serialization_descriptor_deployment_platform(
        handle: *mut c_void,
    ) -> u64;
    pub fn mpsgraph_executable_serialization_descriptor_set_deployment_platform(
        handle: *mut c_void,
        raw_value: u64,
    ) -> bool;
    pub fn mpsgraph_executable_serialization_descriptor_minimum_deployment_target_len(
        handle: *mut c_void,
    ) -> usize;
    pub fn mpsgraph_executable_serialization_descriptor_copy_minimum_deployment_target(
        handle: *mut c_void,
        out_bytes: *mut u8,
        out_len: usize,
    ) -> bool;
    pub fn mpsgraph_executable_serialization_descriptor_set_minimum_deployment_target(
        handle: *mut c_void,
        value: *const c_char,
    ) -> bool;

    pub fn mpsgraph_graph_compile_with_descriptor(
        graph_handle: *mut c_void,
        device_handle: *mut c_void,
        feed_tensors: *const *mut c_void,
        feed_count: usize,
        flat_shapes: *const usize,
        shape_lengths: *const usize,
        data_types: *const u32,
        target_tensors: *const *mut c_void,
        target_count: usize,
        compilation_descriptor_handle: *mut c_void,
    ) -> *mut c_void;

    pub fn mpsgraph_executable_options(handle: *mut c_void) -> u64;
    pub fn mpsgraph_executable_set_options(handle: *mut c_void, raw_value: u64) -> bool;
    pub fn mpsgraph_executable_feed_tensors(handle: *mut c_void) -> *mut c_void;
    pub fn mpsgraph_executable_target_tensors(handle: *mut c_void) -> *mut c_void;
    pub fn mpsgraph_executable_specialize(
        handle: *mut c_void,
        device_handle: *mut c_void,
        input_type_handles: *const *mut c_void,
        input_type_count: usize,
        compilation_descriptor_handle: *mut c_void,
    ) -> bool;
    pub fn mpsgraph_executable_get_output_types(
        handle: *mut c_void,
        device_handle: *mut c_void,
        input_type_handles: *const *mut c_void,
        input_type_count: usize,
        compilation_descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mpsgraph_executable_run_with_descriptor(
        executable_handle: *mut c_void,
        command_queue_handle: *mut c_void,
        input_handles: *const *mut c_void,
        input_count: usize,
        result_handles: *const *mut c_void,
        result_count: usize,
        execution_descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mpsgraph_executable_run_async_with_descriptor(
        executable_handle: *mut c_void,
        command_queue_handle: *mut c_void,
        input_handles: *const *mut c_void,
        input_count: usize,
        result_handles: *const *mut c_void,
        result_count: usize,
        execution_descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mpsgraph_executable_serialize_package(
        handle: *mut c_void,
        path: *const c_char,
        descriptor_handle: *mut c_void,
    ) -> bool;
    pub fn mpsgraph_executable_new_with_package(
        path: *const c_char,
        compilation_descriptor_handle: *mut c_void,
    ) -> *mut c_void;

    pub fn mpsgraph_graph_arithmetic_unary(
        graph_handle: *mut c_void,
        op: u32,
        tensor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_arithmetic_binary(
        graph_handle: *mut c_void,
        op: u32,
        primary_handle: *mut c_void,
        secondary_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_select(
        graph_handle: *mut c_void,
        predicate_handle: *mut c_void,
        true_handle: *mut c_void,
        false_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_relu_gradient(
        graph_handle: *mut c_void,
        gradient_handle: *mut c_void,
        source_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_sigmoid_gradient(
        graph_handle: *mut c_void,
        gradient_handle: *mut c_void,
        source_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_softmax_gradient(
        graph_handle: *mut c_void,
        gradient_handle: *mut c_void,
        source_handle: *mut c_void,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_leaky_relu_scalar(
        graph_handle: *mut c_void,
        tensor_handle: *mut c_void,
        alpha: f64,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_leaky_relu_tensor(
        graph_handle: *mut c_void,
        tensor_handle: *mut c_void,
        alpha_tensor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_leaky_relu_gradient(
        graph_handle: *mut c_void,
        gradient_handle: *mut c_void,
        source_handle: *mut c_void,
        alpha_tensor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_reduction_axis(
        graph_handle: *mut c_void,
        op: u32,
        tensor_handle: *mut c_void,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_reduction_axes(
        graph_handle: *mut c_void,
        op: u32,
        tensor_handle: *mut c_void,
        axes: *const usize,
        axes_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_concat_pair(
        graph_handle: *mut c_void,
        first_handle: *mut c_void,
        second_handle: *mut c_void,
        dimension: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_concat_tensors(
        graph_handle: *mut c_void,
        tensor_handles: *const *mut c_void,
        tensor_count: usize,
        dimension: isize,
        interleave: bool,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_split_sizes(
        graph_handle: *mut c_void,
        tensor_handle: *mut c_void,
        split_sizes: *const usize,
        split_count: usize,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_split_sizes_tensor(
        graph_handle: *mut c_void,
        tensor_handle: *mut c_void,
        split_sizes_tensor_handle: *mut c_void,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_split_num(
        graph_handle: *mut c_void,
        tensor_handle: *mut c_void,
        num_splits: usize,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_stack(
        graph_handle: *mut c_void,
        tensor_handles: *const *mut c_void,
        tensor_count: usize,
        axis: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_pad(
        graph_handle: *mut c_void,
        tensor_handle: *mut c_void,
        padding_mode: isize,
        left_padding: *const isize,
        left_padding_len: usize,
        right_padding: *const isize,
        right_padding_len: usize,
        constant_value: f64,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_top_k(
        graph_handle: *mut c_void,
        source_handle: *mut c_void,
        k: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_top_k_tensor(
        graph_handle: *mut c_void,
        source_handle: *mut c_void,
        k_tensor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
}
