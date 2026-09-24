use apple_metal::{resource_options, MetalBuffer, MetalDevice, MetalTensor};
use apple_mpsgraph::{data_type, data_type_bits, data_type_size, Error, Feed, Graph, TensorData};
use core::ffi::{c_char, c_void};
use core::ptr;

type Id = *mut c_void;

#[link(name = "objc")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Id;
    fn objc_msgSend();
}

macro_rules! msg_send {
    ($receiver:expr, $selector:expr $(, $arg:expr => $arg_ty:ty)* ; -> $ret:ty) => {
        core::mem::transmute::<unsafe extern "C" fn(), unsafe extern "C" fn(Id, Id $(, $arg_ty)*) -> $ret>(
            objc_msgSend,
        )($receiver, sel_registerName($selector.as_ptr()) $(, $arg)*)
    };
}

const MTL_TENSOR_DATA_TYPE_FLOAT32: isize = 3;
const MTL_TENSOR_USAGE_COMPUTE: usize = 1;
const MTL_TENSOR_USAGE_MACHINE_LEARNING: usize = 1 << 2;

unsafe fn tensor_extents(values: &[isize]) -> Id {
    let extents = msg_send!(objc_getClass(c"MTLTensorExtents".as_ptr()), c"alloc"; -> Id);
    msg_send!(
        extents,
        c"initWithRank:values:",
        values.len() => usize,
        values.as_ptr() => *const isize;
        -> Id
    )
}

unsafe fn release(object: Id) {
    msg_send!(object, c"release"; -> ());
}

unsafe fn replace_tensor_values(tensor: Id, extents: &[isize], values: &[f32]) {
    let origin = tensor_extents(&vec![0; extents.len()]);
    let dimensions = tensor_extents(extents);
    let strides = extents
        .iter()
        .scan(1_isize, |stride, extent| {
            let current = *stride;
            *stride *= extent;
            Some(current)
        })
        .collect::<Vec<_>>();
    let strides = tensor_extents(&strides);
    msg_send!(
        tensor,
        c"replaceSliceOrigin:sliceDimensions:withBytes:strides:",
        origin => Id,
        dimensions => Id,
        values.as_ptr().cast::<c_void>() => *const c_void,
        strides => Id;
        -> ()
    );
    release(strides);
    release(dimensions);
    release(origin);
}

fn metal_tensor(
    device: &MetalDevice,
    shape: &[usize],
    usage: usize,
    values: &[f32],
) -> MetalTensor {
    let extents = shape
        .iter()
        .rev()
        .map(|extent| isize::try_from(*extent).expect("extent"))
        .collect::<Vec<_>>();
    unsafe {
        let descriptor = msg_send!(objc_getClass(c"MTLTensorDescriptor".as_ptr()), c"new"; -> Id);
        let dimensions = tensor_extents(&extents);
        msg_send!(descriptor, c"setDimensions:", dimensions => Id; -> ());
        msg_send!(descriptor, c"setDataType:", MTL_TENSOR_DATA_TYPE_FLOAT32 => isize; -> ());
        msg_send!(descriptor, c"setUsage:", usage => usize; -> ());
        let mut error: Id = ptr::null_mut();
        let tensor = msg_send!(
            device.as_ptr(),
            c"newTensorWithDescriptor:error:",
            descriptor => Id,
            &raw mut error => *mut Id;
            -> Id
        );
        release(dimensions);
        release(descriptor);
        assert!(!tensor.is_null(), "newTensorWithDescriptor failed");
        replace_tensor_values(tensor, &extents, values);
        MetalTensor::from_raw(tensor)
    }
}

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn buffer_with(device: &MetalDevice, bytes: &[u8]) -> MetalBuffer {
    let buffer = device
        .new_buffer(bytes.len().max(4), resource_options::STORAGE_MODE_SHARED)
        .expect("buffer");
    unsafe { buffer.write_bytes(0, bytes) }.expect("write buffer");
    buffer
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn macos_major() -> u32 {
    let output = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()
        .expect("sw_vers");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .split('.')
        .next()
        .and_then(|major| major.parse().ok())
        .expect("macOS major version")
}

#[test]
fn data_type_table_matches_the_sdk_encoding() {
    let all = [
        (data_type::FLOAT32, 32),
        (data_type::FLOAT16, 16),
        (data_type::BFLOAT16, 16),
        (data_type::COMPLEX_FLOAT16, 32),
        (data_type::COMPLEX_FLOAT32, 64),
        (data_type::COMPLEX_BFLOAT16, 32),
        (data_type::INT2, 2),
        (data_type::INT4, 4),
        (data_type::INT8, 8),
        (data_type::INT16, 16),
        (data_type::INT32, 32),
        (data_type::INT64, 64),
        (data_type::UINT2, 2),
        (data_type::UINT4, 4),
        (data_type::UINT8, 8),
        (data_type::UINT16, 16),
        (data_type::UINT32, 32),
        (data_type::UINT64, 64),
        (data_type::BOOL, 8),
        (data_type::UNORM8, 8),
        (data_type::FLOAT8_E4M3, 8),
        (data_type::FLOAT8_E5M2, 8),
        (data_type::FLOAT8_E8M0, 8),
        (data_type::FLOAT4_E2M1, 4),
    ];
    for (raw, bits) in all {
        assert_eq!(data_type_bits(raw), Some(bits), "{raw:#x}");
        assert_eq!(usize::try_from(raw & 0xFFFF), Ok(bits), "{raw:#x}");
        let size = (bits % 8 == 0).then_some(bits / 8);
        assert_eq!(data_type_size(raw), size, "{raw:#x}");
    }
    assert_eq!(data_type_bits(data_type::INVALID), None);
    assert_eq!(data_type_bits(0x1234_5678), None);
}

#[test]
fn from_buffer_needs_room_for_the_whole_shape() {
    let device = device();
    let small = buffer_with(&device, &[0; 4]);
    assert_eq!(
        TensorData::from_buffer(&small, &[1024, 1024], data_type::FLOAT32).err(),
        Some(Error::BufferTooSmall {
            required: 4_194_304,
            length: 4,
        })
    );
    assert_eq!(
        TensorData::from_buffer(&small, &[usize::MAX, 2], data_type::FLOAT32).err(),
        Some(Error::Overflow)
    );
    assert_eq!(
        TensorData::from_buffer(&small, &[1], 0x1234).err(),
        Some(Error::UnsupportedDataType(0x1234))
    );
    let values: Vec<f32> = (1_u8..=15).map(f32::from).collect();
    let exact = buffer_with(&device, &f32_bytes(&values));
    assert_eq!(exact.length(), 60);
    let data = TensorData::from_buffer(&exact, &[5, 3], data_type::FLOAT32).expect("packed rows");
    assert_eq!(data.shape(), vec![5, 3]);
    assert_eq!(data.byte_len(), Ok(60));
    assert_eq!(data.read_f32().expect("read"), values);
    assert!(TensorData::from_buffer(&exact, &[4, 4], data_type::FLOAT32).is_err());
}

#[test]
fn sub_byte_tensor_data_is_packed_across_the_array() {
    let device = device();
    let packed = [0x21_u8, 0x43, 0x65];
    assert!(TensorData::from_bytes(&device, &[0; 4], &[2, 3], data_type::INT4).is_none());
    let data =
        TensorData::from_bytes(&device, &packed, &[2, 3], data_type::INT4).expect("int4 data");
    assert_eq!(data.data_type(), data_type::INT4);
    assert_eq!(data.byte_len(), Ok(3));
    assert_eq!(data.read_bytes().expect("read"), packed.to_vec());
    let pairs = TensorData::from_bytes(&device, &[0b1110_0100], &[4], data_type::UINT2)
        .expect("uint2 data");
    assert_eq!(pairs.byte_len(), Ok(1));
    assert_eq!(pairs.read_bytes().expect("read"), vec![0b1110_0100]);
}

#[test]
fn wide_and_complex_types_round_trip() {
    let device = device();
    let bf16: Vec<u8> = [0x3F80_u16, 0x4000, 0xBF80, 0x0000]
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let data = TensorData::from_bytes(&device, &bf16, &[4], data_type::BFLOAT16).expect("bf16");
    assert_eq!(data.byte_len(), Ok(8));
    assert_eq!(data.read_bytes().expect("read"), bf16);
    let complex = f32_bytes(&[1.0, -1.0, 0.5, 2.0]);
    let data = TensorData::from_bytes(&device, &complex, &[2], data_type::COMPLEX_FLOAT32)
        .expect("complex f32");
    assert_eq!(data.byte_len(), Ok(16));
    assert_eq!(data.read_bytes().expect("read"), complex);
}

#[test]
fn float8_tensor_data_needs_macos_27() {
    let device = device();
    let bytes = [0x38_u8, 0x40, 0xB8, 0x00];
    let data = TensorData::from_bytes(&device, &bytes, &[4], data_type::FLOAT8_E4M3);
    if macos_major() >= 27 {
        let data = data.expect("float8 on macOS 27");
        assert_eq!(data.byte_len(), Ok(4));
        assert_eq!(data.read_bytes().expect("read"), bytes.to_vec());
    } else {
        assert!(data.is_none());
    }
}

#[test]
fn oversized_dimensions_fail_instead_of_trapping() {
    let device = device();
    let graph = Graph::new().expect("graph");
    assert!(graph
        .placeholder(Some(&[usize::MAX]), data_type::FLOAT32, None)
        .is_err());
    assert!(graph
        .constant_scalar_shaped(1.0, &[usize::MAX, 0], data_type::FLOAT32)
        .is_err());
    assert!(TensorData::from_bytes(&device, &[], &[usize::MAX, 0], data_type::FLOAT32).is_none());
    assert!(graph.placeholder(Some(&[2]), 0x1234, None).is_err());
    let input = graph
        .placeholder(Some(&[2, 3]), data_type::FLOAT32, None)
        .expect("placeholder");
    let doubled = graph.addition(&input, &input, None).expect("addition");
    let data = TensorData::from_f32_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("data");
    let results = graph
        .run(&[Feed::new(&input, &data)], &[&doubled])
        .expect("run");
    assert_eq!(
        results[0].read_f32().expect("read"),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
}

#[test]
fn metal_tensors_alias_as_tensor_data_on_macos_26() {
    if macos_major() < 26 {
        return;
    }
    let device = device();
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let usage = MTL_TENSOR_USAGE_COMPUTE | MTL_TENSOR_USAGE_MACHINE_LEARNING;
    let tensor = metal_tensor(&device, &[2, 3], usage, &values);
    let data = TensorData::from_tensor(&tensor).expect("tensor data from MTLTensor");
    assert_eq!(data.shape(), vec![2, 3]);
    assert_eq!(data.data_type(), data_type::FLOAT32);
    assert_eq!(data.byte_len(), Ok(24));
    assert_eq!(data.read_f32().expect("read"), values.to_vec());

    let updated = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    unsafe { replace_tensor_values(tensor.as_ptr(), &[3, 2], &updated) };
    assert_eq!(data.read_f32().expect("read alias"), updated.to_vec());

    let compute_only = metal_tensor(&device, &[2, 3], MTL_TENSOR_USAGE_COMPUTE, &values);
    assert!(TensorData::from_tensor(&compute_only).is_none());
}

#[test]
fn reads_refuse_destinations_shorter_than_the_tensor() {
    let device = device();
    let data = TensorData::from_f32_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("data");
    let mut destination = vec![0xA5_u8; 32];
    let short = unsafe {
        apple_mpsgraph::ffi::mpsgraph_tensor_data_read_bytes(
            data.as_ptr(),
            destination.as_mut_ptr().cast(),
            20,
        )
    };
    assert!(!short);
    assert!(destination.iter().all(|byte| *byte == 0xA5));
    let exact = unsafe {
        apple_mpsgraph::ffi::mpsgraph_tensor_data_read_bytes(
            data.as_ptr(),
            destination.as_mut_ptr().cast(),
            24,
        )
    };
    assert!(exact);
    assert_eq!(
        &destination[..24],
        f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).as_slice()
    );
    assert!(destination[24..].iter().all(|byte| *byte == 0xA5));
}
