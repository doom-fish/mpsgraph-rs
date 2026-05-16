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
}
