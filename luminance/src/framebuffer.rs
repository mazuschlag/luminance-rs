use std::fmt;

use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::framebuffer::{Framebuffer as FramebufferBackend, FramebufferBackBuffer};
use crate::context::GraphicsContext;
use crate::texture::{Dim2, Dimensionable, Sampler, TextureError};

pub struct Framebuffer<B, D, CS, DS>
where
  B: ?Sized + FramebufferBackend<D>,
  D: Dimensionable,
  CS: ColorSlot<B, D>,
  DS: DepthSlot<B, D>,
{
  pub(crate) repr: B::FramebufferRepr,
  color_slot: CS::ColorTextures,
  depth_slot: DS::DepthTexture,
}

impl<B, D, CS, DS> Drop for Framebuffer<B, D, CS, DS>
where
  B: ?Sized + FramebufferBackend<D>,
  D: Dimensionable,
  CS: ColorSlot<B, D>,
  DS: DepthSlot<B, D>,
{
  fn drop(&mut self) {
    unsafe { B::destroy_framebuffer(&mut self.repr) }
  }
}

impl<B, D, CS, DS> Framebuffer<B, D, CS, DS>
where
  B: ?Sized + FramebufferBackend<D>,
  D: Dimensionable,
  CS: ColorSlot<B, D>,
  DS: DepthSlot<B, D>,
{
  pub fn new<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self, FramebufferError>
  where
    C: GraphicsContext<Backend = B>,
  {
    unsafe {
      let mut repr = ctx
        .backend()
        .new_framebuffer::<CS, DS>(size, mipmaps, &sampler)?;
      let color_slot = CS::reify_color_textures(ctx, size, mipmaps, &sampler, &mut repr, 0)?;
      let depth_slot = DS::reify_depth_texture(ctx, size, mipmaps, &sampler, &mut repr)?;

      let repr = B::validate_framebuffer(repr)?;

      Ok(Framebuffer {
        repr,
        color_slot,
        depth_slot,
      })
    }
  }

  pub fn size(&self) -> D::Size {
    unsafe { B::framebuffer_size(&self.repr) }
  }

  pub fn color_slot(&self) -> &CS::ColorTextures {
    &self.color_slot
  }

  pub fn depth_slot(&self) -> &DS::DepthTexture {
    &self.depth_slot
  }
}

impl<B> Framebuffer<B, Dim2, (), ()>
where
  B: ?Sized + FramebufferBackend<Dim2> + FramebufferBackBuffer,
{
  pub fn back_buffer<C>(
    ctx: &mut C,
    size: <Dim2 as Dimensionable>::Size,
  ) -> Result<Self, FramebufferError>
  where
    C: GraphicsContext<Backend = B>,
  {
    unsafe { ctx.backend().back_buffer(size) }.map(|repr| Framebuffer {
      repr,
      color_slot: (),
      depth_slot: (),
    })
  }
}

/// Framebuffer error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FramebufferError {
  /// Texture error.
  ///
  /// This happen while creating / associating the color / depth slots.
  TextureError(TextureError),
  /// Incomplete error.
  ///
  /// This happens when finalizing the construction of the framebuffer.
  Incomplete(IncompleteReason),
}

impl fmt::Display for FramebufferError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      FramebufferError::TextureError(ref e) => write!(f, "framebuffer texture error: {}", e),

      FramebufferError::Incomplete(ref e) => write!(f, "incomplete framebuffer: {}", e),
    }
  }
}

impl From<TextureError> for FramebufferError {
  fn from(e: TextureError) -> Self {
    FramebufferError::TextureError(e)
  }
}

impl From<IncompleteReason> for FramebufferError {
  fn from(e: IncompleteReason) -> Self {
    FramebufferError::Incomplete(e)
  }
}

/// Reason a framebuffer is incomplete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IncompleteReason {
  /// Incomplete framebuffer.
  Undefined,
  /// Incomplete attachment (color / depth).
  IncompleteAttachment,
  /// An attachment was missing.
  MissingAttachment,
  /// Incomplete draw buffer.
  IncompleteDrawBuffer,
  /// Incomplete read buffer.
  IncompleteReadBuffer,
  /// Unsupported.
  Unsupported,
  /// Incomplete multisample configuration.
  IncompleteMultisample,
  /// Incomplete layer targets.
  IncompleteLayerTargets,
}

impl fmt::Display for IncompleteReason {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      IncompleteReason::Undefined => write!(f, "incomplete reason"),
      IncompleteReason::IncompleteAttachment => write!(f, "incomplete attachment"),
      IncompleteReason::MissingAttachment => write!(f, "missing attachment"),
      IncompleteReason::IncompleteDrawBuffer => write!(f, "incomplete draw buffer"),
      IncompleteReason::IncompleteReadBuffer => write!(f, "incomplete read buffer"),
      IncompleteReason::Unsupported => write!(f, "unsupported"),
      IncompleteReason::IncompleteMultisample => write!(f, "incomplete multisample"),
      IncompleteReason::IncompleteLayerTargets => write!(f, "incomplete layer targets"),
    }
  }
}
