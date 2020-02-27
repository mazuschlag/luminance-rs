//! Buffer API.

use crate::backend::buffer::{Buffer as BufferBackend, BufferSlice as BufferSliceBackend};
use crate::context::GraphicsContext;

use std::fmt;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Buffer<S, T>
where
  S: BufferBackend<T>,
{
  pub(crate) repr: S::BufferRepr,
  _t: PhantomData<T>,
}

impl<S, T> Drop for Buffer<S, T>
where
  S: BufferBackend<T>,
{
  fn drop(&mut self) {
    unsafe { <S as BufferBackend<T>>::destroy_buffer(&mut self.repr) };
  }
}

impl<S, T> Buffer<S, T>
where
  S: BufferBackend<T>,
{
  pub fn new<C>(ctx: &mut C, len: usize) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
  {
    let repr = unsafe { ctx.backend().new_buffer(len)? };

    Ok(Buffer {
      repr,
      _t: PhantomData,
    })
  }

  pub fn from_slice<C, X>(ctx: &mut C, slice: X) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
    X: AsRef<[T]>,
  {
    let repr = unsafe { ctx.backend().from_slice(slice)? };

    Ok(Buffer {
      repr,
      _t: PhantomData,
    })
  }

  pub fn repeat<C>(ctx: &mut C, len: usize, value: T) -> Result<Self, BufferError>
  where
    C: GraphicsContext<Backend = S>,
    T: Copy,
  {
    let repr = unsafe { ctx.backend().repeat(len, value)? };

    Ok(Buffer {
      repr,
      _t: PhantomData,
    })
  }

  pub fn at(&self, i: usize) -> Option<T>
  where
    T: Copy,
  {
    unsafe { S::at(&self.repr, i) }
  }

  pub fn whole(&self) -> Vec<T>
  where
    T: Copy,
  {
    unsafe { S::whole(&self.repr) }
  }

  pub fn set(&mut self, i: usize, x: T) -> Result<(), BufferError>
  where
    T: Copy,
  {
    unsafe { S::set(&mut self.repr, i, x) }
  }

  pub fn write_whole(&mut self, values: &[T]) -> Result<(), BufferError> {
    unsafe { S::write_whole(&mut self.repr, values) }
  }

  pub fn clear(&mut self, x: T) -> Result<(), BufferError>
  where
    T: Copy,
  {
    unsafe { S::clear(&mut self.repr, x) }
  }

  #[inline(always)]
  pub fn len(&self) -> usize {
    unsafe { S::len(&self.repr) }
  }

  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

impl<S, T> Buffer<S, T>
where
  S: BufferSliceBackend<T>,
{
  pub fn slice(&mut self) -> Result<BufferSlice<S, T>, BufferError> {
    unsafe {
      S::slice_buffer(&mut self.repr).map(|slice| BufferSlice {
        slice,
        _a: PhantomData,
      })
    }
  }
}

/// Buffer errors.
#[derive(Debug, Eq, PartialEq)]
pub enum BufferError {
  /// Overflow when setting a value with a specific index.
  ///
  /// Contains the index and the size of the buffer.
  Overflow { index: usize, buffer_len: usize },

  /// Too few values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooFewValues {
    provided_len: usize,
    buffer_len: usize,
  },

  /// Too many values were passed to fill a buffer.
  ///
  /// Contains the number of passed value and the size of the buffer.
  TooManyValues {
    provided_len: usize,
    buffer_len: usize,
  },

  /// Mapping the buffer failed.
  MapFailed,
}

impl fmt::Display for BufferError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      BufferError::Overflow { index, buffer_len } => write!(
        f,
        "buffer overflow (index = {}, size = {})",
        index, buffer_len
      ),

      BufferError::TooFewValues {
        provided_len,
        buffer_len,
      } => write!(
        f,
        "too few values passed to the buffer (nb = {}, size = {})",
        provided_len, buffer_len
      ),

      BufferError::TooManyValues {
        provided_len,
        buffer_len,
      } => write!(
        f,
        "too many values passed to the buffer (nb = {}, size = {})",
        provided_len, buffer_len
      ),

      BufferError::MapFailed => write!(f, "buffer mapping failed"),
    }
  }
}

#[derive(Debug)]
pub struct BufferSlice<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  slice: S::SliceRepr,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, S, T> Drop for BufferSlice<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  fn drop(&mut self) {
    {
      unsafe { S::destroy_buffer_slice(&mut self.slice) };
    };
  }
}

impl<'a, S, T> BufferSlice<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], BufferError> {
    unsafe { S::obtain_slice(&self.slice) }
  }
}

#[derive(Debug)]
pub struct BufferSliceMut<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  slice: S::SliceMutRepr,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, S, T> Drop for BufferSliceMut<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  fn drop(&mut self) {
    unsafe { S::destroy_buffer_slice_mut(&mut self.slice) };
  }
}

impl<'a, S, T> BufferSliceMut<'a, S, T>
where
  S: BufferSliceBackend<T>,
{
  pub fn as_slice_mut(&mut self) -> Result<&mut [T], BufferError> {
    unsafe { S::obtain_slice_mut(&mut self.slice) }
  }
}
