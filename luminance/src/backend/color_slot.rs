use crate::backend::framebuffer::{Framebuffer, FramebufferError};
use crate::backend::texture::{Dimensionable, Layerable, Sampler, Texture as TextureBackend};
use crate::context::GraphicsContext;
use crate::pixel::{ColorPixel, PixelFormat, RenderablePixel};

use crate::texture::Texture;

pub trait ColorSlot<B, L, D>
where
  B: ?Sized + Framebuffer<L, D>,
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type ColorTextures;

  fn color_formats() -> Vec<PixelFormat>;

  fn reify_color_textures<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
    framebuffer: &mut B::FramebufferRepr,
    attachment_index: usize,
  ) -> Result<Self::ColorTextures, FramebufferError>
  where
    C: GraphicsContext<Backend = B>;
}

impl<B, L, D> ColorSlot<B, L, D> for ()
where
  B: ?Sized + Framebuffer<L, D>,
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
{
  type ColorTextures = ();

  fn color_formats() -> Vec<PixelFormat> {
    Vec::new()
  }

  fn reify_color_textures<C>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: &Sampler,
    _: &mut B::FramebufferRepr,
    _: usize,
  ) -> Result<Self::ColorTextures, FramebufferError>
  where
    C: GraphicsContext<Backend = B>,
  {
    Ok(())
  }
}

impl<B, L, D, P> ColorSlot<B, L, D> for P
where
  B: ?Sized + Framebuffer<L, D> + TextureBackend<L, D, P>,
  L: Layerable,
  D: Dimensionable,
  D::Size: Copy,
  Self: ColorPixel + RenderablePixel,
{
  type ColorTextures = Texture<B, L, D, P>;

  fn color_formats() -> Vec<PixelFormat> {
    vec![P::pixel_format()]
  }

  fn reify_color_textures<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: &Sampler,
    framebuffer: &mut B::FramebufferRepr,
    attachment_index: usize,
  ) -> Result<Self::ColorTextures, FramebufferError>
  where
    C: GraphicsContext<Backend = B>,
  {
    let texture = Texture::new(ctx, size, mipmaps, sampler.clone())?;
    unsafe { B::attach_color_texture(framebuffer, &texture.repr, attachment_index)? };

    Ok(texture)
  }
}

macro_rules! impl_color_slot_tuple {
  ($($pf:ident),*) => {
    impl<B, L, D, $($pf),*> ColorSlot<B, L, D> for ($($pf),*)
    where
      B: ?Sized + Framebuffer<L, D> + $(TextureBackend<L, D, $pf> +)*,
      L: Layerable,
      D: Dimensionable,
      D::Size: Copy,
      $(
        $pf: ColorPixel + RenderablePixel
      ),*
    {
      type ColorTextures = ($(Texture<B, L, D, $pf>),*);

      fn color_formats() -> Vec<PixelFormat> {
        vec![$($pf::pixel_format()),*]
      }

      fn reify_color_textures<C>(
        ctx: &mut C,
        size: D::Size,
        mipmaps: usize,
        sampler: &Sampler,
        framebuffer: &mut B::FramebufferRepr,
        attachment_index: usize,
      ) -> Result<Self::ColorTextures, FramebufferError>
      where
        C: GraphicsContext<Backend = B>, {
          Ok(
            ($(<$pf as ColorSlot<B, L, D>>::reify_color_textures(ctx, size, mipmaps, sampler, framebuffer, attachment_index + 1)?),*)
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
