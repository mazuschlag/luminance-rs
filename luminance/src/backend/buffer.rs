//! Buffer backend interface.
//!
//! This interface defines the low-level API buffers must implement to be usable.

use crate::buffer::BufferError;
use crate::linear::{M22, M33, M44};

pub unsafe trait BufferBase {
  /// The inner representation of the buffer for this backend.
  type BufferRepr;
}

pub unsafe trait Buffer<T>: BufferBase {
  /// Create a new buffer with a given number of uninitialized elements.
  unsafe fn new_buffer(&mut self, len: usize) -> Result<Self::BufferRepr, BufferError>;

  unsafe fn destroy_buffer(buffer: &mut Self::BufferRepr);

  unsafe fn len(buffer: &Self::BufferRepr) -> usize;

  /// Create a new buffer from a slice.
  unsafe fn from_slice<S>(&mut self, slice: S) -> Result<Self::BufferRepr, BufferError>
  where
    S: AsRef<[T]>;

  unsafe fn repeat(&mut self, len: usize, value: T) -> Result<Self::BufferRepr, BufferError>
  where
    T: Copy;

  unsafe fn at(buffer: &Self::BufferRepr, i: usize) -> Option<T>
  where
    T: Copy;

  unsafe fn whole(buffer: &Self::BufferRepr) -> Vec<T>
  where
    T: Copy;

  unsafe fn set(buffer: &mut Self::BufferRepr, i: usize, x: T) -> Result<(), BufferError>
  where
    T: Copy;

  unsafe fn write_whole(buffer: &mut Self::BufferRepr, values: &[T]) -> Result<(), BufferError>;

  unsafe fn clear(buffer: &mut Self::BufferRepr, x: T) -> Result<(), BufferError>
  where
    T: Copy;
}

pub unsafe trait BufferSlice<T>: Buffer<T> {
  type SliceRepr;

  type SliceMutRepr;

  unsafe fn slice_buffer(buffer: &Self::BufferRepr) -> Result<Self::SliceRepr, BufferError>;

  unsafe fn slice_buffer_mut(
    buffer: &mut Self::BufferRepr,
  ) -> Result<Self::SliceMutRepr, BufferError>;

  unsafe fn destroy_buffer_slice(slice: &mut Self::SliceRepr);

  unsafe fn destroy_buffer_slice_mut(slice: &mut Self::SliceMutRepr);

  unsafe fn obtain_slice(slice: &Self::SliceRepr) -> Result<&[T], BufferError>;

  unsafe fn obtain_slice_mut(slice: &mut Self::SliceMutRepr) -> Result<&mut [T], BufferError>;
}

pub unsafe trait UniformBlock {}

unsafe impl UniformBlock for u8 {}
unsafe impl UniformBlock for u16 {}
unsafe impl UniformBlock for u32 {}

unsafe impl UniformBlock for i8 {}
unsafe impl UniformBlock for i16 {}
unsafe impl UniformBlock for i32 {}

unsafe impl UniformBlock for f32 {}
unsafe impl UniformBlock for f64 {}

unsafe impl UniformBlock for bool {}

unsafe impl UniformBlock for M22 {}
unsafe impl UniformBlock for M33 {}
unsafe impl UniformBlock for M44 {}

unsafe impl UniformBlock for [u8; 2] {}
unsafe impl UniformBlock for [u16; 2] {}
unsafe impl UniformBlock for [u32; 2] {}

unsafe impl UniformBlock for [i8; 2] {}
unsafe impl UniformBlock for [i16; 2] {}
unsafe impl UniformBlock for [i32; 2] {}

unsafe impl UniformBlock for [f32; 2] {}
unsafe impl UniformBlock for [f64; 2] {}

unsafe impl UniformBlock for [bool; 2] {}

unsafe impl UniformBlock for [u8; 3] {}
unsafe impl UniformBlock for [u16; 3] {}
unsafe impl UniformBlock for [u32; 3] {}

unsafe impl UniformBlock for [i8; 3] {}
unsafe impl UniformBlock for [i16; 3] {}
unsafe impl UniformBlock for [i32; 3] {}

unsafe impl UniformBlock for [f32; 3] {}
unsafe impl UniformBlock for [f64; 3] {}

unsafe impl UniformBlock for [bool; 3] {}

unsafe impl UniformBlock for [u8; 4] {}
unsafe impl UniformBlock for [u16; 4] {}
unsafe impl UniformBlock for [u32; 4] {}

unsafe impl UniformBlock for [i8; 4] {}
unsafe impl UniformBlock for [i16; 4] {}
unsafe impl UniformBlock for [i32; 4] {}

unsafe impl UniformBlock for [f32; 4] {}
unsafe impl UniformBlock for [f64; 4] {}

unsafe impl UniformBlock for [bool; 4] {}

unsafe impl<T> UniformBlock for [T] where T: UniformBlock {}

macro_rules! impl_uniform_block_tuple {
  ($( $t:ident ),*) => {
    unsafe impl<$($t),*> UniformBlock for ($($t),*) where $($t: UniformBlock),* {}
  }
}

impl_uniform_block_tuple!(A, B);
impl_uniform_block_tuple!(A, B, C);
impl_uniform_block_tuple!(A, B, C, D);
impl_uniform_block_tuple!(A, B, C, D, E);
impl_uniform_block_tuple!(A, B, C, D, E, F);
impl_uniform_block_tuple!(A, B, C, D, E, F, G);
impl_uniform_block_tuple!(A, B, C, D, E, F, G, H);
impl_uniform_block_tuple!(A, B, C, D, E, F, G, H, I);
impl_uniform_block_tuple!(A, B, C, D, E, F, G, H, I, J);
