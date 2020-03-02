//! Texture backend.

use crate::pixel::Pixel;
use crate::texture::{Dimensionable, GenMipmaps, Sampler, TextureError};

/// The base texture trait.
pub unsafe trait TextureBase {
  type TextureRepr;
}

pub unsafe trait Texture<D, P>: TextureBase
where
  D: Dimensionable,
  P: Pixel,
{
  unsafe fn new_texture(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self::TextureRepr, TextureError>;

  unsafe fn destroy_texture(texture: &mut Self::TextureRepr);

  unsafe fn mipmaps(texture: &Self::TextureRepr) -> usize;

  unsafe fn clear_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError>;

  unsafe fn clear(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError>;

  unsafe fn upload_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError>;

  unsafe fn upload(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError>;

  unsafe fn upload_part_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError>;

  unsafe fn upload_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError>;

  unsafe fn get_raw_texels(
    texture: &Self::TextureRepr,
    size: D::Size,
  ) -> Result<Vec<P::RawEncoding>, TextureError>
  where
    P::RawEncoding: Copy + Default;
}
