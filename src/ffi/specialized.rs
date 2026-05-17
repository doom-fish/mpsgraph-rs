use core::ffi::{c_char, c_void};

unsafe extern "C" {
    pub fn mpsgraph_object_retain(handle: *mut c_void) -> *mut c_void;
    pub fn mpsgraph_operation_as_variable(handle: *mut c_void) -> *mut c_void;
    pub fn mpsgraph_variable_op_shape_len(handle: *mut c_void) -> usize;
    pub fn mpsgraph_variable_op_copy_shape(handle: *mut c_void, out_shape: *mut isize);
    pub fn mpsgraph_variable_op_data_type(handle: *mut c_void) -> u32;

    pub fn mpsgraph_convolution3d_descriptor_new(
        stride_in_x: usize,
        stride_in_y: usize,
        stride_in_z: usize,
        dilation_rate_in_x: usize,
        dilation_rate_in_y: usize,
        dilation_rate_in_z: usize,
        groups: usize,
        padding_left: usize,
        padding_right: usize,
        padding_top: usize,
        padding_bottom: usize,
        padding_front: usize,
        padding_back: usize,
        padding_style: usize,
        data_layout: usize,
        weights_layout: usize,
    ) -> *mut c_void;
    pub fn mpsgraph_depthwise_convolution2d_descriptor_new(
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
        weights_layout: usize,
    ) -> *mut c_void;
    pub fn mpsgraph_depthwise_convolution3d_descriptor_new(
        strides: *const usize,
        strides_len: usize,
        dilation_rates: *const usize,
        dilation_rates_len: usize,
        padding_values: *const usize,
        padding_values_len: usize,
        padding_style: usize,
        channel_dimension_index: isize,
    ) -> *mut c_void;
    pub fn mpsgraph_fft_descriptor_new(
        inverse: bool,
        scaling_mode: usize,
        round_to_odd_hermitean: bool,
    ) -> *mut c_void;
    pub fn mpsgraph_im_to_col_descriptor_new(
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
        data_layout: usize,
    ) -> *mut c_void;
    pub fn mpsgraph_pooling4d_descriptor_new(
        kernel_sizes: *const usize,
        kernel_sizes_len: usize,
        strides: *const usize,
        strides_len: usize,
        dilation_rates: *const usize,
        dilation_rates_len: usize,
        padding_values: *const usize,
        padding_values_len: usize,
        padding_style: usize,
        ceil_mode: bool,
        include_zero_pad_to_average: bool,
        return_indices_mode: usize,
        return_indices_data_type: u32,
    ) -> *mut c_void;
    pub fn mpsgraph_sparse_descriptor_new(storage_type: u64, data_type: u32) -> *mut c_void;
    pub fn mpsgraph_stencil_descriptor_new(
        reduction_mode: usize,
        offsets: *const isize,
        offsets_len: usize,
        strides: *const usize,
        strides_len: usize,
        dilation_rates: *const usize,
        dilation_rates_len: usize,
        explicit_padding: *const usize,
        explicit_padding_len: usize,
        boundary_mode: isize,
        padding_style: usize,
        padding_constant: f32,
    ) -> *mut c_void;

    pub fn mpsgraph_graph_convolution3d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        weights_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_convolution_transpose2d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        weights_tensor: *mut c_void,
        output_shape: *const usize,
        output_shape_len: usize,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_cumulative_sum(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axis: isize,
        exclusive: bool,
        reverse: bool,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_depthwise_convolution2d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        weights_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_depthwise_convolution3d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        weights_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_fast_fourier_transform(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axes: *const usize,
        axes_len: usize,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_im_to_col(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_band_part(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        num_lower: isize,
        num_upper: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_softmax_cross_entropy(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        labels_tensor: *mut c_void,
        axis: isize,
        reduction_type: u64,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_matrix_inverse(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_variable_data(
        graph_handle: *mut c_void,
        bytes: *const c_void,
        byte_len: usize,
        shape: *const usize,
        shape_len: usize,
        data_type: u32,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_read_variable(
        graph_handle: *mut c_void,
        variable_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_assign_variable(
        graph_handle: *mut c_void,
        variable_tensor: *mut c_void,
        value_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_non_maximum_suppression(
        graph_handle: *mut c_void,
        boxes_tensor: *mut c_void,
        scores_tensor: *mut c_void,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_non_zero_indices(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_one_hot(
        graph_handle: *mut c_void,
        indices_tensor: *mut c_void,
        depth: usize,
        data_type: u32,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_stochastic_gradient_descent(
        graph_handle: *mut c_void,
        learning_rate_tensor: *mut c_void,
        values_tensor: *mut c_void,
        gradient_tensor: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_max_pooling4d(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_max_pooling4d_return_indices(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_quantize(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        scale: f64,
        zero_point: f64,
        data_type: u32,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_dequantize(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        scale: f64,
        zero_point: f64,
        data_type: u32,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_resize(
        graph_handle: *mut c_void,
        images_tensor: *mut c_void,
        size: *const usize,
        size_len: usize,
        mode: usize,
        center_result: bool,
        align_corners: bool,
        layout: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_resize_nearest(
        graph_handle: *mut c_void,
        images_tensor: *mut c_void,
        size_tensor: *mut c_void,
        nearest_rounding_mode: usize,
        center_result: bool,
        align_corners: bool,
        layout: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_sample_grid(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        coordinate_tensor: *mut c_void,
        layout: usize,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: isize,
        sampling_mode: usize,
        constant_value: f64,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_scatter_nd(
        graph_handle: *mut c_void,
        updates_tensor: *mut c_void,
        indices_tensor: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        batch_dimensions: usize,
        mode: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_scatter(
        graph_handle: *mut c_void,
        updates_tensor: *mut c_void,
        indices_tensor: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        axis: isize,
        mode: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_scatter_along_axis(
        graph_handle: *mut c_void,
        axis: isize,
        updates_tensor: *mut c_void,
        indices_tensor: *mut c_void,
        shape: *const usize,
        shape_len: usize,
        mode: isize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_sort(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axis: isize,
        descending: bool,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_arg_sort(
        graph_handle: *mut c_void,
        tensor: *mut c_void,
        axis: isize,
        descending: bool,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_sparse_tensor_with_descriptor(
        graph_handle: *mut c_void,
        descriptor_handle: *mut c_void,
        tensor_handles: *const *mut c_void,
        tensor_count: usize,
        shape: *const usize,
        shape_len: usize,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_stencil(
        graph_handle: *mut c_void,
        source_tensor: *mut c_void,
        weights_tensor: *mut c_void,
        descriptor_handle: *mut c_void,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn mpsgraph_graph_topk_gradient(
        graph_handle: *mut c_void,
        gradient_tensor: *mut c_void,
        source_tensor: *mut c_void,
        k: usize,
        name: *const c_char,
    ) -> *mut c_void;

    pub fn mpsgraph_execution_descriptor_wait_for_event(
        handle: *mut c_void,
        event_handle: *mut c_void,
        value: u64,
    ) -> bool;
    pub fn mpsgraph_execution_descriptor_signal_event(
        handle: *mut c_void,
        event_handle: *mut c_void,
        execution_stage: u64,
        value: u64,
    ) -> bool;
    pub fn mpsgraph_executable_execution_descriptor_wait_for_event(
        handle: *mut c_void,
        event_handle: *mut c_void,
        value: u64,
    ) -> bool;
    pub fn mpsgraph_executable_execution_descriptor_signal_event(
        handle: *mut c_void,
        event_handle: *mut c_void,
        execution_stage: u64,
        value: u64,
    ) -> bool;
}
