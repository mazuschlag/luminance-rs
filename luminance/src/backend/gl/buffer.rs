//! OpenGL buffer implementation.

use gl;
use gl::types::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;
use std::slice;

use crate::backend::buffer::{Buffer, BufferBase, BufferError, BufferSlice as BufferSliceBackend};
use crate::backend::gl::state::{Bind, GLState};
use crate::backend::gl::GL;

/// OpenGL buffer.
#[derive(Clone)]
pub struct RawBuffer {
  pub(crate) handle: GLuint,
  pub(crate) bytes: usize,
  pub(crate) len: usize,
  state: Rc<RefCell<GLState>>,
}

unsafe impl BufferBase for GL {
  type BufferRepr = RawBuffer;
}

unsafe impl<T> Buffer<T> for GL {
  unsafe fn new_buffer(&mut self, len: usize) -> Result<Self::BufferRepr, BufferError> {
    let mut buffer: GLuint = 0;
    let bytes = mem::size_of::<T>() * len;

    // generate a buffer and force binding the handle; this prevent side-effects from previous bound
    // resources to prevent binding the buffer
    gl::GenBuffers(1, &mut buffer);
    self
      .state
      .borrow_mut()
      .bind_array_buffer(buffer, Bind::Forced);

    gl::BufferData(
      gl::ARRAY_BUFFER,
      bytes as isize,
      ptr::null(),
      gl::STREAM_DRAW,
    );

    Ok(RawBuffer {
      handle: buffer,
      bytes,
      len,
      state: self.state.clone(),
    })
  }

  unsafe fn destroy_buffer(buffer: &mut Self::BufferRepr) {
    buffer.state.borrow_mut().unbind_buffer(buffer.handle);
    gl::DeleteBuffers(1, &buffer.handle);
  }

  unsafe fn len(buffer: &Self::BufferRepr) -> usize {
    buffer.len
  }

  unsafe fn from_slice<S>(&mut self, slice: S) -> Result<Self::BufferRepr, BufferError>
  where
    S: AsRef<[T]>,
  {
    let mut buffer: GLuint = 0;
    let slice = slice.as_ref();
    let len = slice.len();
    let bytes = mem::size_of::<T>() * len;

    gl::GenBuffers(1, &mut buffer);
    self
      .state
      .borrow_mut()
      .bind_array_buffer(buffer, Bind::Cached);
    gl::BufferData(
      gl::ARRAY_BUFFER,
      bytes as isize,
      slice.as_ptr() as *const c_void,
      gl::STREAM_DRAW,
    );

    Ok(RawBuffer {
      handle: buffer,
      bytes,
      len,
      state: self.state.clone(),
    })
  }

  unsafe fn repeat(&mut self, len: usize, value: T) -> Result<Self::BufferRepr, BufferError>
  where
    T: Copy,
  {
    //let mut buf = self.new_buffer(len)?;
    let mut buf = <Self as Buffer<T>>::new_buffer(self, len)?;
    Self::clear(&mut buf, value)?;
    Ok(buf)
  }

  unsafe fn at(buffer: &Self::BufferRepr, i: usize) -> Option<T>
  where
    T: Copy,
  {
    if i >= buffer.len {
      None
    } else {
      buffer
        .state
        .borrow_mut()
        .bind_array_buffer(buffer.handle, Bind::Cached);
      let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *const T;
      let x = *ptr.add(i);
      let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

      Some(x)
    }
  }

  unsafe fn whole(buffer: &Self::BufferRepr) -> Vec<T>
  where
    T: Copy,
  {
    buffer
      .state
      .borrow_mut()
      .bind_array_buffer(buffer.handle, Bind::Cached);
    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *mut T;
    let values = Vec::from_raw_parts(ptr, buffer.len, buffer.len);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    values
  }

  unsafe fn set(buffer: &mut Self::BufferRepr, i: usize, x: T) -> Result<(), BufferError>
  where
    T: Copy,
  {
    if i >= buffer.len {
      Err(BufferError::Overflow {
        index: i,
        buffer_len: buffer.len,
      })
    } else {
      buffer
        .state
        .borrow_mut()
        .bind_array_buffer(buffer.handle, Bind::Cached);
      let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::WRITE_ONLY) as *mut T;
      *ptr.add(i) = x;
      let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

      Ok(())
    }
  }

  unsafe fn write_whole(buffer: &mut Self::BufferRepr, values: &[T]) -> Result<(), BufferError> {
    let len = values.len();
    let in_bytes = len * mem::size_of::<T>();

    // generate warning and recompute the proper number of bytes to copy
    let real_bytes = match in_bytes.cmp(&buffer.bytes) {
      Ordering::Less => {
        return Err(BufferError::TooFewValues {
          provided_len: len,
          buffer_len: buffer.len,
        })
      }

      Ordering::Greater => {
        return Err(BufferError::TooManyValues {
          provided_len: len,
          buffer_len: buffer.len,
        })
      }

      _ => in_bytes,
    };

    buffer
      .state
      .borrow_mut()
      .bind_array_buffer(buffer.handle, Bind::Cached);
    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::WRITE_ONLY);
    ptr::copy_nonoverlapping(values.as_ptr() as *const c_void, ptr, real_bytes);
    let _ = gl::UnmapBuffer(gl::ARRAY_BUFFER);

    Ok(())
  }

  unsafe fn clear(buffer: &mut Self::BufferRepr, x: T) -> Result<(), BufferError>
  where
    T: Copy,
  {
    Self::write_whole(buffer, &vec![x; buffer.len])
  }
}

pub struct BufferSlice<T> {
  buffer: RawBuffer,
  ptr: *const T,
}

pub struct BufferSliceMut<T> {
  buffer: RawBuffer,
  ptr: *mut T,
}

unsafe impl<T> BufferSliceBackend<T> for GL {
  type SliceRepr = BufferSlice<T>;

  type SliceMutRepr = BufferSliceMut<T>;

  unsafe fn slice_buffer(buffer: &Self::BufferRepr) -> Result<Self::SliceRepr, BufferError> {
    buffer
      .state
      .borrow_mut()
      .bind_array_buffer(buffer.handle, Bind::Cached);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_ONLY) as *const T;
    let buffer = buffer.clone();

    if ptr.is_null() {
      Err(BufferError::MapFailed)
    } else {
      Ok(BufferSlice { buffer, ptr })
    }
  }

  unsafe fn slice_buffer_mut(
    buffer: &mut Self::BufferRepr,
  ) -> Result<Self::SliceMutRepr, BufferError> {
    buffer
      .state
      .borrow_mut()
      .bind_array_buffer(buffer.handle, Bind::Cached);

    let ptr = gl::MapBuffer(gl::ARRAY_BUFFER, gl::READ_WRITE) as *mut T;
    let buffer = buffer.clone();

    if ptr.is_null() {
      Err(BufferError::MapFailed)
    } else {
      Ok(BufferSliceMut { buffer, ptr })
    }
  }

  unsafe fn destroy_buffer_slice(slice: &mut Self::SliceRepr) {
    slice
      .buffer
      .state
      .borrow_mut()
      .bind_array_buffer(slice.buffer.handle, Bind::Cached);
    gl::UnmapBuffer(gl::ARRAY_BUFFER);
  }

  unsafe fn destroy_buffer_slice_mut(slice: &mut Self::SliceMutRepr) {
    slice
      .buffer
      .state
      .borrow_mut()
      .bind_array_buffer(slice.buffer.handle, Bind::Cached);
    gl::UnmapBuffer(gl::ARRAY_BUFFER);
  }

  unsafe fn obtain_slice(slice: &Self::SliceRepr) -> Result<&[T], BufferError> {
    Ok(slice::from_raw_parts(slice.ptr, slice.buffer.len))
  }

  unsafe fn obtain_slice_mut(slice: &mut Self::SliceMutRepr) -> Result<&mut [T], BufferError> {
    Ok(slice::from_raw_parts_mut(slice.ptr, slice.buffer.len))
  }
}
