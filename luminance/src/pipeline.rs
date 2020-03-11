use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::pipeline::{
  Pipeline as PipelineBackend, PipelineBase, PipelineBuffer, PipelineTexture,
};
use crate::buffer::Buffer;
use crate::context::GraphicsContext;
use crate::framebuffer::Framebuffer;
use crate::pixel::Pixel;
use crate::shading_gate::ShadingGate;
use crate::texture::Dimensionable;
use crate::texture::Texture;

use std::fmt;
use std::marker::PhantomData;

#[derive(Debug)]
pub enum PipelineError {}

impl fmt::Display for PipelineError {
  fn fmt(&self, _: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    Ok(())
  }
}

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
  pub clear_color: [f32; 4],
  pub clear_color_enabled: bool,
  pub clear_depth_enabled: bool,
  pub viewport: Viewport,
  pub srgb_enabled: bool,
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

pub struct Pipeline<'a, B>
where
  B: ?Sized + PipelineBase,
{
  repr: B::PipelineRepr,
  _phantom: PhantomData<&'a mut ()>,
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
  pub fn new(ctx: &'a mut C) -> Self {
    PipelineGate { ctx }
  }

  pub fn pipeline<D, CS, DS, F>(
    &mut self,
    framebuffer: &Framebuffer<C::Backend, D, CS, DS>,
    pipeline_state: &PipelineState,
    f: F,
  ) -> Result<(), PipelineError>
  where
    C::Backend: FramebufferBackend<D> + PipelineBackend<D>,
    D: Dimensionable,
    CS: ColorSlot<C::Backend, D>,
    DS: DepthSlot<C::Backend, D>,
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
        _phantom: PhantomData,
      })?
    };
    let shading_gate = ShadingGate { ctx: self.ctx };

    f(pipeline, shading_gate);
    Ok(())
  }
}

pub struct BufferBinding<T> {
  binding: u32,
  _phantom: PhantomData<*const T>,
}

impl<T> BufferBinding<T> {
  pub fn binding(self) -> u32 {
    self.binding
  }
}

pub struct BoundBuffer<'a, B, T>
where
  B: PipelineBuffer<T>,
{
  pub(crate) repr: B::BoundBufferRepr,
  _phantom: PhantomData<&'a T>,
}

impl<'a, B, T> BoundBuffer<'a, B, T>
where
  B: PipelineBuffer<T>,
{
  pub fn binding(&self) -> BufferBinding<T> {
    let binding = unsafe { B::buffer_binding(&self.repr) };
    BufferBinding {
      binding,
      _phantom: PhantomData,
    }
  }
}

pub struct TextureBinding<D, S> {
  binding: u32,
  _phantom: PhantomData<*const (D, S)>,
}

impl<D, S> TextureBinding<D, S> {
  pub fn binding(self) -> u32 {
    self.binding
  }
}

pub struct BoundTexture<'a, B, D, P>
where
  B: PipelineTexture<D, P>,
  D: Dimensionable,
  P: Pixel,
{
  pub(crate) repr: B::BoundTextureRepr,
  _phantom: PhantomData<&'a ()>,
}

impl<'a, B, D, P> BoundTexture<'a, B, D, P>
where
  B: PipelineTexture<D, P>,
  D: Dimensionable,
  P: Pixel,
{
  pub fn binding(&self) -> TextureBinding<D, P::SamplerType> {
    let binding = unsafe { B::texture_binding(&self.repr) };
    TextureBinding {
      binding,
      _phantom: PhantomData,
    }
  }
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
    B: PipelineBuffer<T>,
  {
    unsafe {
      B::bind_buffer(&self.repr, &buffer.repr).map(|repr| BoundBuffer {
        repr,
        _phantom: PhantomData,
      })
    }
  }

  pub fn bind_texture<D, P>(
    &'a self,
    texture: &'a Texture<B, D, P>,
  ) -> Result<BoundTexture<'a, B, D, P>, PipelineError>
  where
    B: PipelineTexture<D, P>,
    D: Dimensionable,
    P: Pixel,
  {
    unsafe {
      B::bind_texture(&self.repr, &texture.repr).map(|repr| BoundTexture {
        repr,
        _phantom: PhantomData,
      })
    }
  }
}
