use crate::api::framebuffer::Framebuffer;
use crate::api::shading_gate::ShadingGate;
use crate::api::texture::Texture;
use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::pipeline::{Bound as BoundBackend, Pipeline as PipelineBackend, PipelineError};
use crate::backend::texture::{Dimensionable, Layerable, Texture as TextureBackend};
use crate::context::GraphicsContext;
use crate::pixel::Pixel;

use std::marker::PhantomData;

/// The viewport being part of the [`PipelineState`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Viewport {
  /// The whole viewport is used. The position and dimension of the viewport rectangle are
  /// extracted from the framebuffer.
  Whole,
  /// The viewport is specific and the rectangle area is user-defined.
  Specific {
    /// The lower position on the X axis to start the viewport rectangle at.
    x: u32,
    /// The lower position on the Y axis to start the viewport rectangle at.
    y: u32,
    /// The width of the viewport.
    width: u32,
    /// The height of the viewport.
    height: u32,
  },
}

/// Various customization options for pipelines.
#[derive(Clone, Debug)]
pub struct PipelineState {
  clear_color: [f32; 4],
  clear_color_enabled: bool,
  clear_depth_enabled: bool,
  viewport: Viewport,
  srgb_enabled: bool,
}

impl Default for PipelineState {
  /// Default [`PipelineState`]:
  ///
  /// - Clear color: `[0, 0, 0, 1]`.
  /// - Color is always cleared.
  /// - Depth is always cleared.
  /// - The viewport uses the whole framebuffer’s.
  /// - sRGB encoding is disabled.
  fn default() -> Self {
    PipelineState {
      clear_color: [0., 0., 0., 1.],
      clear_color_enabled: true,
      clear_depth_enabled: true,
      viewport: Viewport::Whole,
      srgb_enabled: false,
    }
  }
}

impl PipelineState {
  /// Create a default [`PipelineState`].
  ///
  /// See the documentation of the [`Default`] for further details.
  pub fn new() -> Self {
    Self::default()
  }

  /// Get the clear color.
  pub fn clear_color(&self) -> [f32; 4] {
    self.clear_color
  }

  /// Set the clear color.
  pub fn set_clear_color(self, clear_color: [f32; 4]) -> Self {
    Self {
      clear_color,
      ..self
    }
  }

  /// Check whether the pipeline’s framebuffer’s color buffers will be cleared.
  pub fn is_clear_color_enabled(&self) -> bool {
    self.clear_color_enabled
  }

  /// Enable clearing color buffers.
  pub fn enable_clear_color(self, clear_color_enabled: bool) -> Self {
    Self {
      clear_color_enabled,
      ..self
    }
  }

  /// Check whether the pipeline’s framebuffer’s depth buffer will be cleared.
  pub fn is_clear_depth_enabled(&self) -> bool {
    self.clear_depth_enabled
  }

  /// Enable clearing depth buffers.
  pub fn enable_clear_depth(self, clear_depth_enabled: bool) -> Self {
    Self {
      clear_depth_enabled,
      ..self
    }
  }

  /// Get the viewport.
  pub fn viewport(&self) -> Viewport {
    self.viewport
  }

  /// Set the viewport.
  pub fn set_viewport(self, viewport: Viewport) -> Self {
    Self { viewport, ..self }
  }

  /// Check whether sRGB linearization is enabled.
  pub fn is_srgb_enabled(&self) -> bool {
    self.srgb_enabled
  }

  /// Enable sRGB linearization.
  pub fn enable_srgb(self, srgb_enabled: bool) -> Self {
    Self {
      srgb_enabled,
      ..self
    }
  }
}

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
    C::Backend: FramebufferBackend<L, D> + PipelineBackend,
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
  B: ?Sized + PipelineBackend,
{
  repr: B::PipelineRepr,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, B> Pipeline<'a, B>
where
  B: PipelineBackend,
{
  pub fn bind_texture<L, D, P>(
    &'a self,
    texture: &'a Texture<B, L, D, P>,
  ) -> Result<BoundTexture<'a, B, L, D, P::SamplerType>, PipelineError>
  where
    B: TextureBackend<L, D, P>,
    L: 'a + Layerable,
    D: 'a + Dimensionable,
    P: 'a + Pixel,
  {
    unsafe {
      B::bind_texture(&self.repr, &texture.repr).map(|repr| BoundTexture {
        repr,
        _t: PhantomData,
      })
    }
  }

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

pub struct BoundTexture<'a, B, L, D, S>
where
  B: PipelineBackend,
{
  repr: B::BoundTextureRepr,
  _t: PhantomData<&'a (L, D, S)>,
}

pub struct Bound<'a, B, T>
where
  B: PipelineBackend + BoundBackend<T>,
{
  repr: B::BoundRepr,
  _t: PhantomData<&'a T>,
}
