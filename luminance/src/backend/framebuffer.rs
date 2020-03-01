//! Framebuffer backend.

use std::fmt;

use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::texture::{
  Dimensionable, Layerable, Sampler, Texture, TextureBase, TextureError,
};
use crate::context::GraphicsContext;
use crate::pixel::{Pixel, PixelFormat};

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

pub unsafe trait Framebuffer<L, D>
where
  Self: TextureBase<L, D>,
  L: Layerable,
  D: Dimensionable,
{
  type FramebufferRepr;

  unsafe fn new_framebuffer(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
  ) -> Result<Self::FramebufferRepr, FramebufferError>;

  unsafe fn destroy_framebuffer(framebuffer: &mut Self::FramebufferRepr);

  unsafe fn attach_color_texture(
    framebuffer: &mut Self::FramebufferRepr,
    texture: &Self::TextureRepr,
    attachment_index: usize,
  ) -> Result<(), FramebufferError>;

  unsafe fn attach_depth_texture(
    framebuffer: &mut Self::FramebufferRepr,
    texture: &Self::TextureRepr,
  ) -> Result<(), FramebufferError>;

  unsafe fn framebuffer_size(framebuffer: &Self::FramebufferRepr) -> D::Size;
}
