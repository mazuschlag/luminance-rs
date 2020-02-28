use gl;
use gl::types::*;
use std::cell::RefCell;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

use crate::gl33::buffer::{BufferSlice, BufferSliceMut, RawBuffer};
use crate::gl33::state::{Bind, GLState};
use crate::gl33::GL33;
use luminance::backend::buffer::{Buffer as _, BufferSlice as BufferSliceBackend};
use luminance::backend::tess::{Tess as TessBackend, TessBuilder as TessBuilderBackend, TessSlice};
use luminance::tess::{Mode, TessError, TessIndex, TessIndexType, TessMapError};
use luminance::vertex::{
  Normalized, Vertex, VertexAttribDesc, VertexAttribDim, VertexAttribType, VertexBufferDesc,
  VertexDesc, VertexInstancing,
};
use luminance::vertex_restart::VertexRestart;

struct VertexBuffer {
  /// Indexed format of the buffer.
  fmt: VertexDesc,
  /// Internal buffer.
  buf: RawBuffer,
}

pub struct TessBuilder {
  vertex_buffers: Vec<VertexBuffer>,
  index_buffer: Option<(RawBuffer, TessIndexType)>,
  restart_index: Option<u32>,
  mode: Mode,
  vert_nb: usize,
  instance_buffers: Vec<VertexBuffer>,
  inst_nb: usize,
}

impl TessBuilder {
  /// Build a tessellation based on a given number of vertices to render by default.
  fn build_tess(self, ctx: &mut GL33, vert_nb: usize, inst_nb: usize) -> Result<Tess, TessError> {
    let mut vao: GLuint = 0;
    let mut gfx_st = ctx.state.borrow_mut();

    unsafe {
      let patch_vert_nb = match self.mode {
        Mode::Patch(nb) => nb,
        _ => 0,
      };

      gl::GenVertexArrays(1, &mut vao);

      // force binding the vertex array so that previously bound vertex arrays (possibly the same
      // handle) don’t prevent us from binding here
      gfx_st.bind_vertex_array(vao, Bind::Forced);

      // add the vertex buffers into the vao
      for vb in &self.vertex_buffers {
        // force binding as it’s meaningful when a vao is bound
        gfx_st.bind_array_buffer(vb.buf.handle, Bind::Forced);
        set_vertex_pointers(&vb.fmt)
      }

      // in case of indexed render, create an index buffer
      if let Some(ref index_buffer) = self.index_buffer {
        // force binding as it’s meaningful when a vao is bound
        gfx_st.bind_element_array_buffer(index_buffer.0.handle, Bind::Forced);
      }

      // add any instance buffers, if any
      for vb in &self.instance_buffers {
        // force binding as it’s meaningful when a vao is bound
        gfx_st.bind_array_buffer(vb.buf.handle, Bind::Forced);
        set_vertex_pointers(&vb.fmt);
      }

      let restart_index = self.restart_index;
      let index_state = self
        .index_buffer
        .map(move |(buffer, index_type)| IndexedDrawState {
          restart_index,
          _buffer: buffer,
          index_type,
        });

      // convert to OpenGL-friendly internals and return
      Ok(Tess {
        mode: opengl_mode(self.mode),
        vert_nb,
        inst_nb,
        patch_vert_nb,
        vao,
        vertex_buffers: self.vertex_buffers,
        instance_buffers: self.instance_buffers,
        index_state,
        state: ctx.state.clone(),
      })
    }
  }

  /// Guess how many vertices there are to render based on the current configuration or fail if
  /// incorrectly configured.
  fn guess_vert_nb_or_fail(&self) -> Result<usize, TessError> {
    if self.vert_nb == 0 {
      // we don’t have an explicit vertex number to render; go and guess!
      if let Some(ref index_buffer) = self.index_buffer {
        // we have an index buffer: just use its size
        Ok(index_buffer.0.len)
      } else {
        // deduce the number of vertices based on the vertex buffers; they all
        // must be of the same length, otherwise it’s an error
        match self.vertex_buffers.len() {
          0 => Err(TessError::AttributelessError(
            "attributeless render with no vertex number".to_owned(),
          )),

          1 => Ok(self.vertex_buffers[0].buf.len),

          _ => {
            let vert_nb = self.vertex_buffers[0].buf.len;
            let incoherent = Self::check_incoherent_buffers(self.vertex_buffers.iter(), vert_nb);

            if incoherent {
              Err(TessError::LengthIncoherency(vert_nb))
            } else {
              Ok(vert_nb)
            }
          }
        }
      }
    } else {
      // we have an explicit number of vertices to render, but we’re gonna check that number actually
      // makes sense
      if let Some(ref index_buffer) = self.index_buffer {
        // we have indices (indirect draw); so we’ll compare to them
        if index_buffer.0.len < self.vert_nb {
          return Err(TessError::Overflow(index_buffer.0.len, self.vert_nb));
        }
      } else {
        let incoherent = Self::check_incoherent_buffers(self.vertex_buffers.iter(), self.vert_nb);

        if incoherent {
          return Err(TessError::LengthIncoherency(self.vert_nb));
        } else if !self.vertex_buffers.is_empty() && self.vertex_buffers[0].buf.len < self.vert_nb {
          return Err(TessError::Overflow(
            self.vertex_buffers[0].buf.len,
            self.vert_nb,
          ));
        }
      }

      Ok(self.vert_nb)
    }
  }

  /// Check whether any vertex buffer is incoherent in its length according to the input length.
  fn check_incoherent_buffers<'b, B>(mut buffers: B, len: usize) -> bool
  where
    B: Iterator<Item = &'b VertexBuffer>,
  {
    !buffers.all(|vb| vb.buf.len == len)
  }

  /// Guess how many instances there are to render based on the current configuration or fail if
  /// incorrectly configured.
  fn guess_inst_nb_or_fail(&self) -> Result<usize, TessError> {
    if self.inst_nb == 0 {
      // we don’t have an explicit instance number to render; go and guess!
      // deduce the number of instances based on the instance buffers; they all must be of the same
      // length, otherwise it’s an error
      match self.instance_buffers.len() {
        0 => {
          // no instance buffer; we we’re not using instance rendering
          Ok(0)
        }

        1 => Ok(self.instance_buffers[0].buf.len),

        _ => {
          let inst_nb = self.instance_buffers[0].buf.len;
          let incoherent = Self::check_incoherent_buffers(self.instance_buffers.iter(), inst_nb);

          if incoherent {
            Err(TessError::LengthIncoherency(inst_nb))
          } else {
            Ok(inst_nb)
          }
        }
      }
    } else {
      // we have an explicit number of instances to render, but we’re gonna check that number
      // actually makes sense
      let incoherent = Self::check_incoherent_buffers(self.instance_buffers.iter(), self.vert_nb);

      if incoherent {
        return Err(TessError::LengthIncoherency(self.inst_nb));
      } else if !self.instance_buffers.is_empty() && self.instance_buffers[0].buf.len < self.inst_nb
      {
        return Err(TessError::Overflow(
          self.instance_buffers[0].buf.len,
          self.inst_nb,
        ));
      }

      Ok(self.inst_nb)
    }
  }
}

unsafe impl TessBuilderBackend for GL33 {
  type TessBuilderRepr = TessBuilder;

  unsafe fn new_tess_builder(&mut self) -> Result<Self::TessBuilderRepr, TessError> {
    Ok(TessBuilder {
      vertex_buffers: Vec::new(),
      index_buffer: None,
      restart_index: None,
      mode: Mode::Point,
      vert_nb: 0,
      instance_buffers: Vec::new(),
      inst_nb: 0,
    })
  }

  unsafe fn add_vertices<V, W>(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    vertices: W,
  ) -> Result<(), TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    let vertices = vertices.as_ref();

    let vb = VertexBuffer {
      fmt: V::vertex_desc(),
      buf: self.from_slice(vertices)?,
    };

    tess_builder.vertex_buffers.push(vb);

    Ok(())
  }

  unsafe fn add_instances<V, W>(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    instances: W,
  ) -> Result<(), TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    let instances = instances.as_ref();

    let vb = VertexBuffer {
      fmt: V::vertex_desc(),
      buf: self.from_slice(instances)?,
    };

    tess_builder.instance_buffers.push(vb);

    Ok(())
  }

  unsafe fn set_indices<T, I>(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    indices: T,
  ) -> Result<(), TessError>
  where
    T: AsRef<[I]>,
    I: TessIndex,
  {
    let indices = indices.as_ref();

    // create a new raw buffer containing the indices and turn it into a vertex buffer
    let buf = self.from_slice(indices)?;

    tess_builder.index_buffer = Some((buf, I::INDEX_TYPE));

    Ok(())
  }

  unsafe fn set_mode(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    mode: Mode,
  ) -> Result<(), TessError> {
    tess_builder.mode = mode;
    Ok(())
  }

  unsafe fn set_vertex_nb(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    nb: usize,
  ) -> Result<(), TessError> {
    tess_builder.vert_nb = nb;
    Ok(())
  }

  unsafe fn set_instance_nb(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    nb: usize,
  ) -> Result<(), TessError> {
    tess_builder.inst_nb = nb;
    Ok(())
  }

  unsafe fn set_primitive_restart_index(
    &mut self,
    tess_builder: &mut Self::TessBuilderRepr,
    index: Option<u32>,
  ) -> Result<(), TessError> {
    tess_builder.restart_index = index;
    Ok(())
  }
}

pub struct Tess {
  mode: GLenum,
  vert_nb: usize,
  inst_nb: usize,
  patch_vert_nb: usize,
  vao: GLenum,
  vertex_buffers: Vec<VertexBuffer>,
  instance_buffers: Vec<VertexBuffer>,
  index_state: Option<IndexedDrawState>,
  state: Rc<RefCell<GLState>>,
}

/// All the extra data required when doing indexed drawing.
struct IndexedDrawState {
  _buffer: RawBuffer,
  restart_index: Option<u32>,
  index_type: TessIndexType,
}

unsafe impl TessBackend for GL33 {
  type TessRepr = Tess;

  unsafe fn build(
    &mut self,
    tess_builder: Self::TessBuilderRepr,
  ) -> Result<Self::TessRepr, TessError> {
    // try to deduce the number of vertices to render if it’s not specified
    let vert_nb = tess_builder.guess_vert_nb_or_fail()?;
    let inst_nb = tess_builder.guess_inst_nb_or_fail()?;
    tess_builder.build_tess(self, vert_nb, inst_nb)
  }

  unsafe fn destroy_tess(tess: &mut Self::TessRepr) {
    tess.state.borrow_mut().unbind_vertex_array();
    gl::DeleteVertexArrays(1, &tess.vao);
  }

  unsafe fn tess_vertices_nb(tess: &Self::TessRepr) -> usize {
    tess.vert_nb
  }

  unsafe fn tess_instances_nb(tess: &Self::TessRepr) -> usize {
    tess.inst_nb
  }

  unsafe fn render(
    tess: &Self::TessRepr,
    start_index: usize,
    vert_nb: usize,
    inst_nb: usize,
  ) -> Result<(), TessError> {
    let vert_nb = vert_nb as GLsizei;
    let inst_nb = inst_nb as GLsizei;

    let mut gfx_st = tess.state.borrow_mut();
    gfx_st.bind_vertex_array(tess.vao, Bind::Cached);

    if tess.mode == gl::PATCHES {
      gfx_st.set_patch_vertex_nb(tess.patch_vert_nb);
    }

    if let Some(index_state) = tess.index_state.as_ref() {
      // indexed render
      let first = (index_state.index_type.bytes() * start_index) as *const c_void;

      if let Some(restart_index) = index_state.restart_index {
        gfx_st.set_vertex_restart(VertexRestart::On);
        gl::PrimitiveRestartIndex(restart_index);
      } else {
        gfx_st.set_vertex_restart(VertexRestart::Off);
      }

      if inst_nb <= 1 {
        gl::DrawElements(
          tess.mode,
          vert_nb,
          index_type_to_glenum(index_state.index_type),
          first,
        );
      } else {
        gl::DrawElementsInstanced(
          tess.mode,
          vert_nb,
          index_type_to_glenum(index_state.index_type),
          first,
          inst_nb,
        );
      }
    } else {
      // direct render
      let first = start_index as GLint;

      if inst_nb <= 1 {
        gl::DrawArrays(tess.mode, first, vert_nb);
      } else {
        gl::DrawArraysInstanced(tess.mode, first, vert_nb, inst_nb);
      }
    }

    Ok(())
  }
}

unsafe impl<T> TessSlice<T> for GL33 {
  type SliceRepr = BufferSlice<T>;

  type SliceMutRepr = BufferSliceMut<T>;

  unsafe fn destroy_tess_slice(slice: &mut Self::SliceRepr) {
    <GL33 as BufferSliceBackend<T>>::destroy_buffer_slice(slice);
  }

  unsafe fn destroy_tess_slice_mut(slice: &mut Self::SliceMutRepr) {
    <GL33 as BufferSliceBackend<T>>::destroy_buffer_slice_mut(slice);
  }

  unsafe fn slice_vertices(tess: &Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: Vertex,
  {
    match tess.vertex_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &tess.vertex_buffers[0];
        let target_fmt = T::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::VertexTypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          GL33::slice_buffer(&vb.buf).map_err(TessMapError::BufferMapError)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn slice_vertices_mut(
    tess: &mut Self::TessRepr,
  ) -> Result<Self::SliceMutRepr, TessMapError>
  where
    T: Vertex,
  {
    match tess.vertex_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &mut tess.vertex_buffers[0];
        let target_fmt = T::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::VertexTypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          GL33::slice_buffer_mut(&mut vb.buf).map_err(TessMapError::BufferMapError)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn slice_indices(tess: &Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: TessIndex,
  {
    match tess.index_state {
      Some(IndexedDrawState {
        ref _buffer,
        ref index_type,
        ..
      }) => {
        let target_fmt = T::INDEX_TYPE;

        if *index_type != target_fmt {
          Err(TessMapError::IndexTypeMismatch(*index_type, target_fmt))
        } else {
          GL33::slice_buffer(_buffer).map_err(TessMapError::BufferMapError)
        }
      }

      None => Err(TessMapError::ForbiddenAttributelessMapping),
    }
  }

  unsafe fn slice_indices_mut(tess: &mut Self::TessRepr) -> Result<Self::SliceMutRepr, TessMapError>
  where
    T: TessIndex,
  {
    match tess.index_state {
      Some(IndexedDrawState {
        ref mut _buffer,
        ref index_type,
        ..
      }) => {
        let target_fmt = T::INDEX_TYPE;

        if *index_type != target_fmt {
          Err(TessMapError::IndexTypeMismatch(*index_type, target_fmt))
        } else {
          GL33::slice_buffer_mut(_buffer).map_err(TessMapError::BufferMapError)
        }
      }

      None => Err(TessMapError::ForbiddenAttributelessMapping),
    }
  }

  unsafe fn slice_instances(tess: &Self::TessRepr) -> Result<Self::SliceRepr, TessMapError>
  where
    T: Vertex,
  {
    match tess.instance_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &tess.instance_buffers[0];
        let target_fmt = T::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::VertexTypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          GL33::slice_buffer(&vb.buf).map_err(TessMapError::BufferMapError)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn slice_instances_mut(
    tess: &mut Self::TessRepr,
  ) -> Result<Self::SliceMutRepr, TessMapError>
  where
    T: Vertex,
  {
    match tess.instance_buffers.len() {
      0 => Err(TessMapError::ForbiddenAttributelessMapping),

      1 => {
        let vb = &mut tess.instance_buffers[0];
        let target_fmt = T::vertex_desc(); // costs a bit

        if vb.fmt != target_fmt {
          Err(TessMapError::VertexTypeMismatch(vb.fmt.clone(), target_fmt))
        } else {
          GL33::slice_buffer_mut(&mut vb.buf).map_err(TessMapError::BufferMapError)
        }
      }

      _ => Err(TessMapError::ForbiddenDeinterleavedMapping),
    }
  }

  unsafe fn obtain_slice(slice: &Self::SliceRepr) -> Result<&[T], TessMapError> {
    <GL33 as BufferSliceBackend<T>>::obtain_slice(slice).map_err(TessMapError::BufferMapError)
  }

  unsafe fn obtain_slice_mut(slice: &mut Self::SliceMutRepr) -> Result<&mut [T], TessMapError> {
    <GL33 as BufferSliceBackend<T>>::obtain_slice_mut(slice).map_err(TessMapError::BufferMapError)
  }
}

// Give OpenGL types information on the content of the VBO by setting vertex descriptors and pointers
// to buffer memory.
fn set_vertex_pointers(descriptors: &[VertexBufferDesc]) {
  // this function sets the vertex attribute pointer for the input list by computing:
  //   - The vertex attribute ID: this is the “rank” of the attribute in the input list (order
  //     matters, for short).
  //   - The stride: this is easily computed, since it’s the size (bytes) of a single vertex.
  //   - The offsets: each attribute has a given offset in the buffer. This is computed by
  //     accumulating the size of all previously set attributes.
  let offsets = aligned_offsets(descriptors);
  let vertex_weight = offset_based_vertex_weight(descriptors, &offsets) as GLsizei;

  for (desc, off) in descriptors.iter().zip(offsets) {
    set_component_format(vertex_weight, off, desc);
  }
}

// Compute offsets for all the vertex components according to the alignments provided.
fn aligned_offsets(descriptor: &[VertexBufferDesc]) -> Vec<usize> {
  let mut offsets = Vec::with_capacity(descriptor.len());
  let mut off = 0;

  // compute offsets
  for desc in descriptor {
    let desc = &desc.attrib_desc;
    off = off_align(off, desc.align); // keep the current component descriptor aligned
    offsets.push(off);
    off += component_weight(desc); // increment the offset by the pratical size of the component
  }

  offsets
}

// Align an offset.
#[inline]
fn off_align(off: usize, align: usize) -> usize {
  let a = align - 1;
  (off + a) & !a
}

// Weight in bytes of a vertex component.
fn component_weight(f: &VertexAttribDesc) -> usize {
  dim_as_size(f.dim) as usize * f.unit_size
}

fn dim_as_size(d: VertexAttribDim) -> GLint {
  match d {
    VertexAttribDim::Dim1 => 1,
    VertexAttribDim::Dim2 => 2,
    VertexAttribDim::Dim3 => 3,
    VertexAttribDim::Dim4 => 4,
  }
}

// Weight in bytes of a single vertex, taking into account padding so that the vertex stay correctly
// aligned.
fn offset_based_vertex_weight(descriptors: &[VertexBufferDesc], offsets: &[usize]) -> usize {
  if descriptors.is_empty() || offsets.is_empty() {
    return 0;
  }

  off_align(
    offsets[offsets.len() - 1] + component_weight(&descriptors[descriptors.len() - 1].attrib_desc),
    descriptors[0].attrib_desc.align,
  )
}

// Set the vertex component OpenGL pointers regarding the index of the component (i), the stride
fn set_component_format(stride: GLsizei, off: usize, desc: &VertexBufferDesc) {
  let attrib_desc = &desc.attrib_desc;
  let index = desc.index as GLuint;

  unsafe {
    match attrib_desc.ty {
      VertexAttribType::Floating => {
        gl::VertexAttribPointer(
          index,
          dim_as_size(attrib_desc.dim),
          opengl_sized_type(&attrib_desc),
          gl::FALSE,
          stride,
          ptr::null::<c_void>().add(off),
        );
      }

      VertexAttribType::Integral(Normalized::No)
      | VertexAttribType::Unsigned(Normalized::No)
      | VertexAttribType::Boolean => {
        // non-normalized integrals / booleans
        gl::VertexAttribIPointer(
          index,
          dim_as_size(attrib_desc.dim),
          opengl_sized_type(&attrib_desc),
          stride,
          ptr::null::<c_void>().add(off),
        );
      }

      _ => {
        // normalized integrals
        gl::VertexAttribPointer(
          index,
          dim_as_size(attrib_desc.dim),
          opengl_sized_type(&attrib_desc),
          gl::TRUE,
          stride,
          ptr::null::<c_void>().add(off),
        );
      }
    }

    // set vertex attribute divisor based on the vertex instancing configuration
    let divisor = match desc.instancing {
      VertexInstancing::On => 1,
      VertexInstancing::Off => 0,
    };
    gl::VertexAttribDivisor(index, divisor);

    gl::EnableVertexAttribArray(index);
  }
}

fn opengl_sized_type(f: &VertexAttribDesc) -> GLenum {
  match (f.ty, f.unit_size) {
    (VertexAttribType::Integral(_), 1) => gl::BYTE,
    (VertexAttribType::Integral(_), 2) => gl::SHORT,
    (VertexAttribType::Integral(_), 4) => gl::INT,
    (VertexAttribType::Unsigned(_), 1) | (VertexAttribType::Boolean, 1) => gl::UNSIGNED_BYTE,
    (VertexAttribType::Unsigned(_), 2) => gl::UNSIGNED_SHORT,
    (VertexAttribType::Unsigned(_), 4) => gl::UNSIGNED_INT,
    (VertexAttribType::Floating, 4) => gl::FLOAT,
    _ => panic!("unsupported vertex component format: {:?}", f),
  }
}

fn opengl_mode(mode: Mode) -> GLenum {
  match mode {
    Mode::Point => gl::POINTS,
    Mode::Line => gl::LINES,
    Mode::LineStrip => gl::LINE_STRIP,
    Mode::Triangle => gl::TRIANGLES,
    Mode::TriangleFan => gl::TRIANGLE_FAN,
    Mode::TriangleStrip => gl::TRIANGLE_STRIP,
    Mode::Patch(_) => gl::PATCHES,
  }
}

fn index_type_to_glenum(ty: TessIndexType) -> GLenum {
  match ty {
    TessIndexType::U8 => gl::UNSIGNED_BYTE,
    TessIndexType::U16 => gl::UNSIGNED_SHORT,
    TessIndexType::U32 => gl::UNSIGNED_INT,
  }
}
