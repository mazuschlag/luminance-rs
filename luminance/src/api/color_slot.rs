use crate::backend::texture::{
  Dimensionable, Layerable, Sampler, Texture as TextureBackend, TextureBase, TextureError,
};
use crate::context::GraphicsContext;
use crate::pixel::{ColorPixel, Pixel, PixelFormat, RenderablePixel};

use crate::api::texture::Texture;

pub trait ColorSlot<L, D, B>
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type ColorTextures;

  fn color_formats() -> Vec<PixelFormat>;

  fn reify_textures<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
  ) -> Result<Self::ColorTextures, TextureError>
  where
    B: TextureBase<L, D>,
    C: GraphicsContext<Backend = B>;
}

impl<L, D, B> ColorSlot<L, D, B> for ()
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type ColorTextures = ();

  fn color_formats() -> Vec<PixelFormat> {
    Vec::new()
  }

  fn reify_textures<C>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: &Sampler,
  ) -> Result<Self::ColorTextures, TextureError>
  where
    B: TextureBase<L, D>,
    C: GraphicsContext<Backend = B>,
  {
    Ok(())
  }
}

impl<L, D, B, P> ColorSlot<L, D, B> for P
where
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
  B: TextureBackend<L, D, P>,
  Self: ColorPixel + RenderablePixel,
{
  type ColorTextures = Texture<B, L, D, P>;

  fn color_formats() -> Vec<PixelFormat> {
    vec![P::pixel_format()]
  }

  fn reify_textures<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
  ) -> Result<Self::ColorTextures, TextureError>
  where
    B: TextureBase<L, D>,
    C: GraphicsContext<Backend = B>,
  {
    Texture::new(ctx, size, mipmaps, sampler.clone())
  }
}

macro_rules! impl_color_slot_tuple {
  ($($pf:ident),*) => {
    impl<L, D, B, $($pf),*> ColorSlot<L, D, B> for ($($pf),*)
    where
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      B: $(TextureBackend<L, D, $pf> +)*,
      $(
        $pf: ColorPixel + RenderablePixel
      ),*
    {
      type ColorTextures = ($(Texture<B, L, D, $pf>),*);

      fn color_formats() -> Vec<PixelFormat> {
        vec![$($pf::pixel_format()),*]
      }

      fn reify_textures<C>(
        ctx: &mut C,
        size: D::Size,
        mipmaps: usize,
        sampler: &Sampler,
      ) -> Result<Self::ColorTextures, TextureError>
      where
        B: TextureBase<L, D>,
        C: GraphicsContext<Backend = B>, {
          Ok(
            ($(<$pf as ColorSlot<L, D, B>>::reify_textures(ctx, size, mipmaps, sampler)?),*)
          )
      }
    }
  }
}

macro_rules! impl_color_slot_tuples {
  ($first:ident , $second:ident) => {
    // stop at pairs
    impl_color_slot_tuple!($first, $second);
  };

  ($first:ident , $($pf:ident),*) => {
    // implement the same list without the first type (reduced by one)
    impl_color_slot_tuples!($($pf),*);
    // implement the current list
    impl_color_slot_tuple!($first, $($pf),*);
  };
}

impl_color_slot_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
