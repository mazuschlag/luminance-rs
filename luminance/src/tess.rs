//! Tessellation API.

use std::fmt;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::backend::tess::{
  Tess as TessBackend, TessBuilder as TessBuilderBackend, TessSlice as TessSliceBackend,
};
use crate::buffer::BufferError;
use crate::context::GraphicsContext;
use crate::vertex::Vertex;
use crate::vertex::VertexDesc;

/// Vertices can be connected via several modes.
///
/// Some modes allow for _primitive restart_. Primitive restart is a cool feature that allows to
/// _break_ the building of a primitive to _start over again_. For instance, when making a curve,
/// you can imagine gluing segments next to each other. If at some point, you want to start a new
/// line, you have two choices:
///
///   - Either you stop your draw call and make another one.
///   - Or you just use the _primitive restart_ feature to ask to create another line from scratch.
///
/// That feature is encoded with a special _vertex index_. You can setup the value of the _primitive
/// restart index_ with [`TessBuilder::set_primitive_restart_index`]. Whenever a vertex index is set
/// to the same value as the _primitive restart index_, the value is not interpreted as a vertex
/// index but just a marker / hint to start a new primitive.
#[derive(Copy, Clone, Debug)]
pub enum Mode {
  /// A single point.
  ///
  /// Points are left unconnected from each other and represent a _point cloud_. This is the typical
  /// primitive mode you want to do, for instance, particles rendering.
  Point,
  /// A line, defined by two points.
  ///
  /// Every pair of vertices are connected together to form a straight line.
  Line,
  /// A strip line, defined by at least two points and zero or many other ones.
  ///
  /// The first two vertices create a line, and every new vertex flowing in the graphics pipeline
  /// (starting from the third, then) well extend the initial line, making a curve composed of
  /// several segments.
  ///
  /// > This kind of primitive mode allows the usage of _primitive restart_.
  LineStrip,
  /// A triangle, defined by three points.
  Triangle,
  /// A triangle fan, defined by at least three points and zero or many other ones.
  ///
  /// Such a mode is easy to picture: a cooling fan is a circular shape, with blades.
  /// [`Mode::TriangleFan`] is kind of the same. The first vertex is at the center of the fan, then
  /// the second vertex creates the first edge of the first triangle. Every time you add a new
  /// vertex, a triangle is created by taking the first (center) vertex, the very previous vertex
  /// and the current vertex. By specifying vertices around the center, you actually create a
  /// fan-like shape.
  ///
  /// > This kind of primitive mode allows the usage of _primitive restart_.
  TriangleFan,
  /// A triangle strip, defined by at least three points and zero or many other ones.
  ///
  /// This mode is a bit different from [`Mode::TriangleFan`]. The first two vertices define the
  /// first edge of the first triangle. Then, for each new vertex, a new triangle is created by
  /// taking the very previous vertex and the last to very previous vertex. What it means is that
  /// every time a triangle is created, the next vertex will share the edge that was created to
  /// spawn the previous triangle.
  ///
  /// This mode is useful to create long ribbons / strips of triangles.
  ///
  /// > This kind of primitive mode allows the usage of _primitive restart_.
  TriangleStrip,
  /// A general purpose primitive with _n_ vertices, for use in tessellation shaders.
  /// For example, `Mode::Patch(3)` represents triangle patches, so every three vertices in the
  /// buffer form a patch.
  /// If you want to employ tessellation shaders, this is the only primitive mode you can use.
  Patch(usize),
}

/// Error that can occur while trying to map GPU tessellation to host code.
#[derive(Debug, Eq, PartialEq)]
pub enum TessMapError {
  /// The CPU mapping failed due to buffer errors.
  BufferMapError(BufferError),
  /// Vertex target type is not the same as the one stored in the buffer.
  VertexTypeMismatch(VertexDesc, VertexDesc),
  /// Index target type is not the same as the one stored in the buffer.
  IndexTypeMismatch(TessIndexType, TessIndexType),
  /// The CPU mapping failed because you cannot map an attributeless tessellation since it doesn’t
  /// have any vertex attribute.
  ForbiddenAttributelessMapping,
  /// The CPU mapping failed because currently, mapping deinterleaved buffers is not supported via
  /// a single slice.
  ForbiddenDeinterleavedMapping,
}

impl fmt::Display for TessMapError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      TessMapError::BufferMapError(ref e) => write!(f, "cannot map tessellation buffer: {}", e),
      TessMapError::VertexTypeMismatch(ref a, ref b) => write!(
        f,
        "cannot map tessellation: vertex type mismatch between {:?} and {:?}",
        a, b
      ),
      TessMapError::IndexTypeMismatch(ref a, ref b) => write!(
        f,
        "cannot map tessellation: index type mismatch between {:?} and {:?}",
        a, b
      ),
      TessMapError::ForbiddenAttributelessMapping => {
        f.write_str("cannot map an attributeless buffer")
      }
      TessMapError::ForbiddenDeinterleavedMapping => {
        f.write_str("cannot map a deinterleaved buffer as interleaved")
      }
    }
  }
}

/// Possible errors that might occur when dealing with [`Tess`].
#[derive(Debug)]
pub enum TessError {
  /// Error related to attributeless tessellation and/or render.
  AttributelessError(String),
  /// Length incoherency in vertex, index or instance buffers.
  LengthIncoherency(usize),
  /// Overflow when accessing underlying buffers.
  Overflow(usize, usize),
  /// Internal error ocurring with a buffer.
  InternalBufferError(BufferError),
}

impl From<BufferError> for TessError {
  fn from(e: BufferError) -> Self {
    TessError::InternalBufferError(e)
  }
}

/// Possible tessellation index types.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TessIndexType {
  /// 8-bit unsigned integer.
  U8,
  /// 16-bit unsigned integer.
  U16,
  /// 32-bit unsigned integer.
  U32,
}

impl TessIndexType {
  pub fn bytes(self) -> usize {
    match self {
      TessIndexType::U8 => 1,
      TessIndexType::U16 => 2,
      TessIndexType::U32 => 4,
    }
  }
}

/// Class of tessellation indexes.
///
/// Values which types implement this trait are allowed to be used to index tessellation in *indexed
/// draw commands*.
///
/// You shouldn’t have to worry to much about that trait. Have a look at the current implementors
/// for an exhaustive list of types you can use.
///
/// > Implementing this trait is `unsafe`.
pub unsafe trait TessIndex {
  /// Type of the underlying index.
  ///
  /// You are limited in which types you can use as indexes. Feel free to have a look at the
  /// documentation of the [`TessIndexType`] trait for further information.
  const INDEX_TYPE: TessIndexType;
}

unsafe impl TessIndex for u8 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U8;
}

unsafe impl TessIndex for u16 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U16;
}

unsafe impl TessIndex for u32 {
  const INDEX_TYPE: TessIndexType = TessIndexType::U32;
}

pub struct TessBuilder<'a, C>
where
  C: GraphicsContext,
  C::Backend: TessBuilderBackend,
{
  ctx: &'a mut C,
  repr: <C::Backend as TessBuilderBackend>::TessBuilderRepr,
}

impl<'a, C> TessBuilder<'a, C>
where
  C: GraphicsContext,
  C::Backend: TessBuilderBackend,
{
  pub fn new(ctx: &'a mut C) -> Result<Self, TessError> {
    unsafe {
      ctx
        .backend()
        .new_tess_builder()
        .map(move |repr| TessBuilder { ctx, repr })
    }
  }

  pub fn add_vertices<V, W>(mut self, vertices: W) -> Result<Self, TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    unsafe {
      self
        .ctx
        .backend()
        .add_vertices(&mut self.repr, vertices)
        .map(move |_| self)
    }
  }

  pub fn add_instances<V, W>(mut self, instances: W) -> Result<Self, TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    unsafe {
      self
        .ctx
        .backend()
        .add_instances(&mut self.repr, instances)
        .map(move |_| self)
    }
  }

  pub fn set_indices<T, I>(mut self, indices: T) -> Result<Self, TessError>
  where
    T: AsRef<[I]>,
    I: TessIndex,
  {
    unsafe {
      self
        .ctx
        .backend()
        .set_indices(&mut self.repr, indices)
        .map(move |_| self)
    }
  }

  pub fn set_mode(mut self, mode: Mode) -> Result<Self, TessError> {
    unsafe {
      self
        .ctx
        .backend()
        .set_mode(&mut self.repr, mode)
        .map(move |_| self)
    }
  }

  pub fn set_vertex_nb(mut self, nb: usize) -> Result<Self, TessError> {
    unsafe {
      self
        .ctx
        .backend()
        .set_vertex_nb(&mut self.repr, nb)
        .map(move |_| self)
    }
  }

  pub fn set_instance_nb(mut self, nb: usize) -> Result<Self, TessError> {
    unsafe {
      self
        .ctx
        .backend()
        .set_instance_nb(&mut self.repr, nb)
        .map(move |_| self)
    }
  }

  pub fn set_primitive_restart_index<T>(mut self, index: T) -> Result<Self, TessError>
  where
    T: Into<Option<u32>>,
  {
    unsafe {
      self
        .ctx
        .backend()
        .set_primitive_restart_index(&mut self.repr, index.into())
        .map(move |_| self)
    }
  }
}

impl<'a, C> TessBuilder<'a, C>
where
  C: GraphicsContext,
  C::Backend: TessBuilderBackend + TessBackend,
{
  pub fn build(self) -> Result<Tess<C::Backend>, TessError> {
    unsafe {
      self
        .ctx
        .backend()
        .build(self.repr)
        .map(|repr| Tess { repr })
    }
  }
}

#[derive(Debug)]
pub struct Tess<S>
where
  S: ?Sized + TessBackend,
{
  pub(crate) repr: S::TessRepr,
}

impl<S> Drop for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn drop(&mut self) {
    unsafe { S::destroy_tess(&mut self.repr) };
  }
}

impl<S> Tess<S>
where
  S: ?Sized + TessBackend,
{
  pub fn vert_nb(&self) -> usize {
    unsafe { S::tess_vertices_nb(&self.repr) }
  }

  pub fn inst_nb(&self) -> usize {
    unsafe { S::tess_instances_nb(&self.repr) }
  }

  pub fn slice_vertices<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_vertices(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_vertices_mut<T>(&mut self) -> Result<TessSliceMut<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_vertices_mut(&mut self.repr).map(|repr| TessSliceMut { repr }) }
  }

  pub fn slice_indices<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: TessIndex,
  {
    unsafe { S::slice_indices(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_indices_mut<T>(&mut self) -> Result<TessSliceMut<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: TessIndex,
  {
    unsafe { S::slice_indices_mut(&mut self.repr).map(|repr| TessSliceMut { repr }) }
  }

  pub fn slice_instances<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_instances(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_instances_mut<T>(&mut self) -> Result<TessSliceMut<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_instances_mut(&mut self.repr).map(|repr| TessSliceMut { repr }) }
  }
}

#[derive(Debug)]
pub struct TessSlice<S, T>
where
  S: ?Sized + TessSliceBackend<T>,
{
  repr: S::SliceRepr,
}

impl<S, T> Drop for TessSlice<S, T>
where
  S: ?Sized + TessSliceBackend<T>,
{
  fn drop(&mut self) {
    unsafe { S::destroy_tess_slice(&mut self.repr) };
  }
}

impl<S, T> TessSlice<S, T>
where
  S: ?Sized + TessSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], TessMapError> {
    unsafe { S::obtain_slice(&self.repr) }
  }
}

#[derive(Debug)]
pub struct TessSliceMut<S, T>
where
  S: ?Sized + TessSliceBackend<T>,
{
  repr: S::SliceMutRepr,
}

impl<S, T> Drop for TessSliceMut<S, T>
where
  S: ?Sized + TessSliceBackend<T>,
{
  fn drop(&mut self) {
    unsafe { S::destroy_tess_slice_mut(&mut self.repr) };
  }
}

impl<S, T> TessSliceMut<S, T>
where
  S: ?Sized + TessSliceBackend<T>,
{
  pub fn as_slice_mut(&mut self) -> Result<&mut [T], TessMapError> {
    unsafe { S::obtain_slice_mut(&mut self.repr) }
  }
}

#[derive(Debug)]
pub enum TessViewError {
  IncorrectViewWindow {
    capacity: usize,
    start: usize,
    nb: usize,
  },
}

#[derive(Clone)]
pub struct TessView<'a, S>
where
  S: ?Sized + TessBackend,
{
  /// Tessellation to render.
  pub(crate) tess: &'a Tess<S>,
  /// Start index (vertex) in the tessellation.
  pub(crate) start_index: usize,
  /// Number of vertices to pick from the tessellation.
  pub(crate) vert_nb: usize,
  /// Number of instances to render.
  pub(crate) inst_nb: usize,
}

impl<'a, S> TessView<'a, S>
where
  S: ?Sized + TessBackend,
{
  pub fn one_whole(tess: &'a Tess<S>) -> Self {
    TessView {
      tess,
      start_index: 0,
      vert_nb: tess.vert_nb(),
      inst_nb: tess.inst_nb(),
    }
  }

  pub fn inst_whole(tess: &'a Tess<S>, inst_nb: usize) -> Self {
    TessView {
      tess,
      start_index: 0,
      vert_nb: tess.vert_nb(),
      inst_nb,
    }
  }

  pub fn one_sub(tess: &'a Tess<S>, vert_nb: usize) -> Result<Self, TessViewError> {
    let capacity = tess.vert_nb();

    if vert_nb > capacity {
      return Err(TessViewError::IncorrectViewWindow {
        capacity,
        start: 0,
        nb: vert_nb,
      });
    }

    Ok(TessView {
      tess,
      start_index: 0,
      vert_nb,
      inst_nb: 1,
    })
  }

  pub fn inst_sub(
    tess: &'a Tess<S>,
    vert_nb: usize,
    inst_nb: usize,
  ) -> Result<Self, TessViewError> {
    let capacity = tess.vert_nb();

    if vert_nb > capacity {
      return Err(TessViewError::IncorrectViewWindow {
        capacity,
        start: 0,
        nb: vert_nb,
      });
    }

    Ok(TessView {
      tess,
      start_index: 0,
      vert_nb,
      inst_nb,
    })
  }

  pub fn one_slice(tess: &'a Tess<S>, start: usize, nb: usize) -> Result<Self, TessViewError> {
    let capacity = tess.vert_nb();

    if start > capacity || nb + start > capacity {
      return Err(TessViewError::IncorrectViewWindow {
        capacity,
        start,
        nb,
      });
    }

    Ok(TessView {
      tess,
      start_index: start,
      vert_nb: nb,
      inst_nb: 1,
    })
  }

  pub fn inst_slice(
    tess: &'a Tess<S>,
    start: usize,
    nb: usize,
    inst_nb: usize,
  ) -> Result<Self, TessViewError> {
    let capacity = tess.vert_nb();

    if start > capacity || nb + start > capacity {
      return Err(TessViewError::IncorrectViewWindow {
        capacity,
        start,
        nb,
      });
    }

    Ok(TessView {
      tess,
      start_index: start,
      vert_nb: nb,
      inst_nb,
    })
  }
}

impl<'a, S> From<&'a Tess<S>> for TessView<'a, S>
where
  S: ?Sized + TessBackend,
{
  fn from(tess: &'a Tess<S>) -> Self {
    TessView::one_whole(tess)
  }
}

pub trait SubTess<S, Idx>
where
  S: ?Sized + TessBackend,
{
  /// Slice a tessellation object and yields a [`TessSlice`] according to the index range.
  fn slice(&self, idx: Idx) -> Result<TessView<S>, TessViewError>;

  /// Slice a tesselation object and yields a [`TessSlice`] according to the index range with as
  /// many instances as specified.
  fn inst_slice(&self, idx: Idx, inst_nb: usize) -> Result<TessView<S>, TessViewError>;
}

impl<S> SubTess<S, RangeFull> for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn slice(&self, _: RangeFull) -> Result<TessView<S>, TessViewError> {
    Ok(TessView::one_whole(self))
  }

  fn inst_slice(&self, _: RangeFull, inst_nb: usize) -> Result<TessView<S>, TessViewError> {
    Ok(TessView::inst_whole(self, inst_nb))
  }
}

impl<S> SubTess<S, RangeTo<usize>> for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn slice(&self, to: RangeTo<usize>) -> Result<TessView<S>, TessViewError> {
    TessView::one_sub(self, to.end)
  }

  fn inst_slice(&self, to: RangeTo<usize>, inst_nb: usize) -> Result<TessView<S>, TessViewError> {
    TessView::inst_sub(self, to.end, inst_nb)
  }
}

impl<S> SubTess<S, RangeFrom<usize>> for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn slice(&self, from: RangeFrom<usize>) -> Result<TessView<S>, TessViewError> {
    TessView::one_slice(self, from.start, self.vert_nb() - from.start)
  }

  fn inst_slice(
    &self,
    from: RangeFrom<usize>,
    inst_nb: usize,
  ) -> Result<TessView<S>, TessViewError> {
    TessView::inst_slice(self, from.start, self.vert_nb() - from.start, inst_nb)
  }
}

impl<S> SubTess<S, Range<usize>> for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn slice(&self, range: Range<usize>) -> Result<TessView<S>, TessViewError> {
    TessView::one_slice(self, range.start, range.end - range.start)
  }

  fn inst_slice(&self, range: Range<usize>, inst_nb: usize) -> Result<TessView<S>, TessViewError> {
    TessView::inst_slice(self, range.start, range.end - range.start, inst_nb)
  }
}

impl<S> SubTess<S, RangeInclusive<usize>> for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn slice(&self, range: RangeInclusive<usize>) -> Result<TessView<S>, TessViewError> {
    let start = *range.start();
    let end = *range.end();
    TessView::one_slice(self, start, end - start + 1)
  }

  fn inst_slice(
    &self,
    range: RangeInclusive<usize>,
    inst_nb: usize,
  ) -> Result<TessView<S>, TessViewError> {
    let start = *range.start();
    let end = *range.end();
    TessView::inst_slice(self, start, end - start + 1, inst_nb)
  }
}

impl<S> SubTess<S, RangeToInclusive<usize>> for Tess<S>
where
  S: ?Sized + TessBackend,
{
  fn slice(&self, to: RangeToInclusive<usize>) -> Result<TessView<S>, TessViewError> {
    TessView::one_sub(self, to.end + 1)
  }

  fn inst_slice(
    &self,
    to: RangeToInclusive<usize>,
    inst_nb: usize,
  ) -> Result<TessView<S>, TessViewError> {
    TessView::inst_sub(self, to.end + 1, inst_nb)
  }
}
