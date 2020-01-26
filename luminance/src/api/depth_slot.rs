use crate::backend::texture::{
  Dimensionable, Layerable, Sampler, Texture as TextureBackend, TextureBase, TextureError,
};
use crate::context::GraphicsContext;
use crate::pixel::{DepthPixel, Pixel, PixelFormat};

use crate::api::texture::Texture;

pub trait DepthSlot<L, D, B>
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  /// Texture associated with this color slot.
  type DepthTexture;

  /// Turn a depth slot into a pixel format.
  fn depth_format() -> Option<PixelFormat>;

  /// Reify a raw textures into a depth slot.
  fn reify_texture<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
  ) -> Result<Self::DepthTexture, TextureError>
  where
    C: GraphicsContext<Backend = B>;
}

impl<L, D, B> DepthSlot<L, D, B> for ()
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type DepthTexture = ();

  fn depth_format() -> Option<PixelFormat> {
    None
  }

  fn reify_texture<C>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: &Sampler,
  ) -> Result<Self::DepthTexture, TextureError>
  where
    C: GraphicsContext<Backend = B>,
  {
    Ok(())
  }
}

impl<L, D, B, P> DepthSlot<L, D, B> for P
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
  B: TextureBackend<L, D, P>,
  P: DepthPixel,
{
  type DepthTexture = Texture<B, L, D, P>;

  fn depth_format() -> Option<PixelFormat> {
    Some(P::pixel_format())
  }

  fn reify_texture<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
  ) -> Result<Self::DepthTexture, TextureError>
  where
    C: GraphicsContext<Backend = B>,
  {
    Texture::new(ctx, size, mipmaps, sampler.clone())
  }
}
