use crate::api::framebuffer::Framebuffer;
use crate::api::shading_gate::ShadingGate;
use crate::api::texture::Texture;
use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::pipeline::{
  Bound as BoundBackend, Pipeline as PipelineBackend, PipelineBase, PipelineError, PipelineState,
  Viewport,
};
use crate::backend::texture::{Dimensionable, Layerable, Texture as TextureBackend};
use crate::context::GraphicsContext;
use crate::pixel::Pixel;

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
  pub fn bind<T>(&'a self, resource: &'a T) -> Result<Bound<'a, B, T>, PipelineError>
  where
    B: BoundBackend<T>,
  {
    unsafe {
      B::bind_resource(&self.repr, resource).map(|repr| Bound {
        repr,
        _t: PhantomData,
      })
    }
  }
}

pub struct Bound<'a, B, T>
where
  B: PipelineBase + BoundBackend<T>,
{
  repr: B::BoundRepr,
  _t: PhantomData<&'a T>,
}
