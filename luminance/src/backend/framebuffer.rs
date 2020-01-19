//! Framebuffer backend.

use std::fmt;

use crate::backend::texture::{Dimensionable, Layerable, Texture, TextureError};
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

pub unsafe trait ColorSlot<L, D, P>
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type ColorTextures;

  fn color_formats() -> Vec<PixelFormat>;

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I,
  ) -> Self::ColorTextures
  where
    C: GraphicsContext<Backend = Self>,
    I: Iterator<Item = Self::TextureRepr>;
}

unsafe impl<L, D, B> ColorSlot<L, D, ()> for B
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type ColorTextures = ();

  fn color_formats() -> Vec<PixelFormat> {
    Vec::new()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I,
  ) -> Self::ColorTextures
  where
    C: GraphicsContext<Backend = Self>,
    I: Iterator<Item = Self::TextureRepr>,
  {
  }
}

//pub unsafe trait Framebuffer<L, D, CS, DS> {
//}
