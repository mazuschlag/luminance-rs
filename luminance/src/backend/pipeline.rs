use crate::api::pipeline::PipelineState;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::shading_gate::ShadingGate as ShadingGateBackend;
use crate::backend::texture::{Dimensionable, Layerable};

use std::fmt;

#[derive(Debug)]
pub enum PipelineError {}

impl fmt::Display for PipelineError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    Ok(())
  }
}

pub unsafe trait Pipeline: ShadingGateBackend {
  type PipelineRepr;

  unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError>;

  unsafe fn start_pipeline<L, D>(
    &mut self,
    framebuffer: &Self::FramebufferRepr,
    pipeline_state: &PipelineState,
  ) where
    Self: FramebufferBackend<L, D>,
    L: Layerable,
    D: Dimensionable;
}
