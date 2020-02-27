use crate::backend::buffer::Buffer;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::shading_gate::ShadingGate as ShadingGateBackend;
use crate::backend::texture::{Texture, TextureBase};
use crate::pipeline::{PipelineError, PipelineState};
use crate::pixel::Pixel;
use crate::texture::{Dimensionable, Layerable};

pub unsafe trait PipelineBase: ShadingGateBackend + TextureBase {
  type PipelineRepr;

  unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError>;
}

pub unsafe trait Pipeline<L, D>: PipelineBase + FramebufferBackend<L, D>
where
  L: Layerable,
  D: Dimensionable,
{
  unsafe fn start_pipeline(
    &mut self,
    framebuffer: &Self::FramebufferRepr,
    pipeline_state: &PipelineState,
  );
}

pub unsafe trait PipelineBuffer<T>: PipelineBase + Buffer<T> {
  type BoundBufferRepr;

  unsafe fn bind_buffer(
    pipeline: &Self::PipelineRepr,
    buffer: &Self::BufferRepr,
  ) -> Result<Self::BoundBufferRepr, PipelineError>;
}

pub unsafe trait PipelineTexture<L, D, P>: PipelineBase + Texture<L, D, P>
where
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  type BoundTextureRepr;

  unsafe fn bind_texture(
    pipeline: &Self::PipelineRepr,
    texture: &Self::TextureRepr,
  ) -> Result<Self::BoundTextureRepr, PipelineError>
  where
    L: Layerable,
    D: Dimensionable,
    P: Pixel;
}
