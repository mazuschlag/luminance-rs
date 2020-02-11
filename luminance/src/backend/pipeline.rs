use crate::api::pipeline::PipelineState;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::shading_gate::ShadingGate as ShadingGateBackend;
use crate::backend::texture::{Dimensionable, Layerable, Texture};
use crate::pixel::Pixel;

use std::fmt;

#[derive(Debug)]
pub enum PipelineError {}

impl fmt::Display for PipelineError {
  fn fmt(&self, _: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    Ok(())
  }
}

pub unsafe trait Pipeline: ShadingGateBackend {
  type PipelineRepr;

  type BoundTextureRepr;

  unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError>;

  unsafe fn start_pipeline<L, D>(
    &mut self,
    framebuffer: &Self::FramebufferRepr,
    pipeline_state: &PipelineState,
  ) where
    Self: FramebufferBackend<L, D>,
    L: Layerable,
    D: Dimensionable;

  unsafe fn bind_texture<L, D, P>(
    pipeline: &Self::PipelineRepr,
    texture: &Self::TextureRepr,
  ) -> Result<Self::BoundTextureRepr, PipelineError>
  where
    Self: Texture<L, D, P>,
    L: Layerable,
    D: Dimensionable,
    P: Pixel;
}

pub unsafe trait Bound<T>: Pipeline {
  type BoundRepr;

  unsafe fn bind_resource(
    pipeline: &Self::PipelineRepr,
    resource: &T,
  ) -> Result<Self::BoundRepr, PipelineError>;
}
