//! Framebuffer backend.

use crate::backend::color_slot::ColorSlot;
use crate::backend::depth_slot::DepthSlot;
use crate::backend::texture::TextureBase;
use crate::framebuffer::FramebufferError;
use crate::texture::{Dim2, Dimensionable, Sampler};

pub unsafe trait Framebuffer<D>: TextureBase
where
  D: Dimensionable,
{
  type FramebufferRepr;

  unsafe fn new_framebuffer<CS, DS>(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
  ) -> Result<Self::FramebufferRepr, FramebufferError>
  where
    CS: ColorSlot<Self, D>,
    DS: DepthSlot<Self, D>;

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

  unsafe fn validate_framebuffer(
    framebuffer: Self::FramebufferRepr,
  ) -> Result<Self::FramebufferRepr, FramebufferError>;

  unsafe fn framebuffer_size(framebuffer: &Self::FramebufferRepr) -> D::Size;
}

pub unsafe trait FramebufferBackBuffer: Framebuffer<Dim2> {
  unsafe fn back_buffer(
    &mut self,
    size: <Dim2 as Dimensionable>::Size,
  ) -> Result<Self::FramebufferRepr, FramebufferError>;
}
