//! Tessellation API.

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::backend::tess::{
  Mode, Tess as TessBackend, TessBuilder as TessBuilderBackend, TessError, TessIndex, TessMapError,
  TessSlice as TessSliceBackend,
};
use crate::context::GraphicsContext;
use crate::vertex::Vertex;

#[derive(Debug)]
pub struct TessBuilder<S>
where
  S: TessBuilderBackend,
{
  repr: S::TessBuilderRepr,
}

impl<S> TessBuilder<S>
where
  S: TessBuilderBackend,
{
  pub fn new<C>(ctx: &mut C) -> Result<Self, TessError>
  where
    C: GraphicsContext<Backend = S>,
  {
    unsafe {
      ctx
        .backend()
        .new_tess_builder()
        .map(|repr| TessBuilder { repr })
    }
  }

  pub fn add_vertices<V, W>(mut self, vertices: W) -> Result<Self, TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    unsafe { S::add_vertices(&mut self.repr, vertices).map(move |_| self) }
  }

  pub fn add_instances<V, W>(mut self, instances: W) -> Result<Self, TessError>
  where
    W: AsRef<[V]>,
    V: Vertex,
  {
    unsafe { S::add_instances(&mut self.repr, instances).map(move |_| self) }
  }

  pub fn set_indices<T, I>(mut self, indices: T) -> Result<Self, TessError>
  where
    T: AsRef<[I]>,
    I: TessIndex,
  {
    unsafe { S::set_indices(&mut self.repr, indices).map(move |_| self) }
  }

  pub fn set_mode(mut self, mode: Mode) -> Result<Self, TessError> {
    unsafe { S::set_mode(&mut self.repr, mode).map(move |_| self) }
  }

  pub fn set_vertex_nb(mut self, nb: usize) -> Result<Self, TessError> {
    unsafe { S::set_vertex_nb(&mut self.repr, nb).map(move |_| self) }
  }

  pub fn set_instance_nb(mut self, nb: usize) -> Result<Self, TessError> {
    unsafe { S::set_instance_nb(&mut self.repr, nb).map(move |_| self) }
  }

  pub fn set_primitive_restart_index<T>(mut self, index: T) -> Result<Self, TessError>
  where
    T: Into<Option<u32>>,
  {
    unsafe { S::set_primitive_restart_index(&mut self.repr, index.into()).map(move |_| self) }
  }
}

impl<S> TessBuilder<S>
where
  S: TessBackend,
{
  pub fn build(self) -> Result<Tess<S>, TessError> {
    unsafe { S::build(self.repr).map(|repr| Tess { repr }) }
  }
}

#[derive(Debug)]
pub struct Tess<S>
where
  S: TessBackend,
{
  repr: S::TessRepr,
}

impl<S> Drop for Tess<S>
where
  S: TessBackend,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_tess(&mut self.repr) };
  }
}

impl<S> Tess<S>
where
  S: TessBackend,
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

  pub fn slice_vertices_mut<T>(&mut self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_vertices_mut(&mut self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_indices<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: TessIndex,
  {
    unsafe { S::slice_indices(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_indices_mut<T>(&mut self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: TessIndex,
  {
    unsafe { S::slice_indices_mut(&mut self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_instances<T>(&self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_instances(&self.repr).map(|repr| TessSlice { repr }) }
  }

  pub fn slice_instances_mut<T>(&mut self) -> Result<TessSlice<S, T>, TessMapError>
  where
    S: TessSliceBackend<T>,
    T: Vertex,
  {
    unsafe { S::slice_instances_mut(&mut self.repr).map(|repr| TessSlice { repr }) }
  }
}

#[derive(Debug)]
pub struct TessSlice<S, T>
where
  S: TessSliceBackend<T>,
{
  repr: S::SliceRepr,
}

impl<S, T> Drop for TessSlice<S, T>
where
  S: TessSliceBackend<T>,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_tess_slice(&mut self.repr) };
  }
}

impl<S, T> TessSlice<S, T>
where
  S: TessSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], TessMapError> {
    unsafe { S::obtain_slice(&self.repr) }
  }
}

#[derive(Debug)]
pub struct TessSliceMut<S, T>
where
  S: TessSliceBackend<T>,
{
  repr: S::SliceRepr,
}

impl<S, T> Drop for TessSliceMut<S, T>
where
  S: TessSliceBackend<T>,
{
  fn drop(&mut self) {
    let _ = unsafe { S::destroy_tess_slice(&mut self.repr) };
  }
}

impl<S, T> TessSliceMut<S, T>
where
  S: TessSliceBackend<T>,
{
  pub fn as_slice(&self) -> Result<&[T], TessMapError> {
    unsafe { S::obtain_slice(&self.repr) }
  }

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
  S: TessBackend,
{
  /// Tessellation to render.
  tess: &'a Tess<S>,
  /// Start index (vertex) in the tessellation.
  start_index: usize,
  /// Number of vertices to pick from the tessellation.
  vert_nb: usize,
  /// Number of instances to render.
  inst_nb: usize,
}

impl<'a, S> TessView<'a, S>
where
  S: TessBackend,
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
  S: TessBackend,
{
  fn from(tess: &'a Tess<S>) -> Self {
    TessView::one_whole(tess)
  }
}

pub trait SubTess<S, Idx>
where
  S: TessBackend,
{
  /// Slice a tessellation object and yields a [`TessSlice`] according to the index range.
  fn slice(&self, idx: Idx) -> Result<TessView<S>, TessViewError>;

  /// Slice a tesselation object and yields a [`TessSlice`] according to the index range with as
  /// many instances as specified.
  fn inst_slice(&self, idx: Idx, inst_nb: usize) -> Result<TessView<S>, TessViewError>;
}

impl<S> SubTess<S, RangeFull> for Tess<S>
where
  S: TessBackend,
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
  S: TessBackend,
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
  S: TessBackend,
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
  S: TessBackend,
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
  S: TessBackend,
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
  S: TessBackend,
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
