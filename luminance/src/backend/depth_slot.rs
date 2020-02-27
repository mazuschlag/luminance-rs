use crate::backend::framebuffer::Framebuffer;
use crate::backend::texture::Texture as TextureBackend;
use crate::context::GraphicsContext;
use crate::framebuffer::FramebufferError;
use crate::pixel::{DepthPixel, PixelFormat};
use crate::texture::{Dimensionable, Layerable, Sampler};

use crate::texture::Texture;

pub trait DepthSlot<B, L, D>
where
  B: ?Sized + Framebuffer<L, D>,
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  /// Texture associated with this color slot.
  type DepthTexture;

  /// Turn a depth slot into a pixel format.
  fn depth_format() -> Option<PixelFormat>;

  /// Reify a raw textures into a depth slot.
  fn reify_depth_texture<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
    framebuffer: &mut B::FramebufferRepr,
  ) -> Result<Self::DepthTexture, FramebufferError>
  where
    C: GraphicsContext<Backend = B>;
}

impl<B, L, D> DepthSlot<B, L, D> for ()
where
  B: ?Sized + Framebuffer<L, D>,
  D::Size: Copy,
  L: Layerable,
  D: Dimensionable,
{
  type DepthTexture = ();

  fn depth_format() -> Option<PixelFormat> {
    None
  }

  fn reify_depth_texture<C>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: &Sampler,
    _: &mut B::FramebufferRepr,
  ) -> Result<Self::DepthTexture, FramebufferError>
  where
    C: GraphicsContext<Backend = B>,
  {
    Ok(())
  }
}

impl<B, L, D, P> DepthSlot<B, L, D> for P
where
  B: ?Sized + Framebuffer<L, D> + TextureBackend<L, D, P>,
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
  P: DepthPixel,
{
  type DepthTexture = Texture<B, L, D, P>;

  fn depth_format() -> Option<PixelFormat> {
    Some(P::pixel_format())
  }

  fn reify_depth_texture<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
    framebuffer: &mut B::FramebufferRepr,
  ) -> Result<Self::DepthTexture, FramebufferError>
  where
    C: GraphicsContext<Backend = B>,
  {
    let texture = Texture::new(ctx, size, mipmaps, sampler.clone())?;
    unsafe { B::attach_depth_texture(framebuffer, &texture.repr)? };

    Ok(texture)
  }
}
