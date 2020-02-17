use crate::api::buffer::Buffer;
use crate::api::framebuffer::Framebuffer;
use crate::api::shading_gate::ShadingGate;
use crate::api::texture::Texture;
use crate::backend::buffer::Buffer as BufferBackend;
use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::pipeline::{
  Pipeline as PipelineBackend, PipelineBase, PipelineError, PipelineState,
};
use crate::backend::texture::{Dimensionable, Layerable, Texture as TextureBackend};
use crate::context::GraphicsContext;
use crate::pixel::{Pixel, SamplerType};

use std::marker::PhantomData;

pub struct PipelineGate<'a, C>
where
  C: ?Sized + GraphicsContext,
{
  ctx: &'a mut C,
}

impl<'a, C> PipelineGate<'a, C>
where
  C: ?Sized + GraphicsContext,
{
  pub fn pipeline<L, D, CS, DS, F>(
    &mut self,
    framebuffer: &Framebuffer<C::Backend, L, D, CS, DS>,
    pipeline_state: &PipelineState,
    f: F,
  ) -> Result<(), PipelineError>
  where
    C::Backend: FramebufferBackend<L, D> + PipelineBackend<L, D>,
    L: Layerable,
    D: Dimensionable,
    CS: ColorSlot<C::Backend, L, D>,
    DS: DepthSlot<C::Backend, L, D>,
    F: for<'b> FnOnce(Pipeline<'b, C::Backend>, ShadingGate<'b, C>),
  {
    unsafe {
      self
        .ctx
        .backend()
        .start_pipeline(&framebuffer.repr, pipeline_state);
    }

    let pipeline = unsafe {
      self.ctx.backend().new_pipeline().map(|repr| Pipeline {
        repr,
        _a: PhantomData,
      })?
    };
    let shading_gate = ShadingGate { ctx: self.ctx };

    f(pipeline, shading_gate);
    Ok(())
  }
}

pub struct Pipeline<'a, B>
where
  B: ?Sized + PipelineBase,
{
  repr: B::PipelineRepr,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, B> Pipeline<'a, B>
where
  B: PipelineBase,
{
  pub fn bind_buffer<T>(
    &'a self,
    buffer: &'a Buffer<B, T>,
  ) -> Result<BoundBuffer<'a, B, T>, PipelineError>
  where
    B: BufferBackend<T>,
  {
    unsafe {
      B::bind_buffer(&self.repr, &buffer.repr).map(|repr| BoundBuffer {
        repr,
        _t: PhantomData,
      })
    }
  }

  pub fn bind_texture<L, D, P>(
    &'a self,
    texture: &'a Texture<B, L, D, P>,
  ) -> Result<BoundTexture<'a, B, L, D, P::SamplerType>, PipelineError>
  where
    B: TextureBackend<L, D, P>,
    L: Layerable,
    D: Dimensionable,
    P: Pixel,
  {
    unsafe {
      B::bind_texture::<L, D, P>(&self.repr, &texture.repr).map(|repr| BoundTexture {
        repr,
        _t: PhantomData,
      })
    }
  }
}

pub struct BoundBuffer<'a, B, T>
where
  B: PipelineBase,
{
  pub(crate) repr: B::BoundBufferRepr,
  _t: PhantomData<&'a T>,
}

pub struct BoundTexture<'a, B, L, D, S>
where
  B: PipelineBase,
  L: Layerable,
  D: Dimensionable,
  S: SamplerType,
{
  pub(crate) repr: B::BoundTextureRepr,
  _t: PhantomData<&'a (L, D, S)>,
}
