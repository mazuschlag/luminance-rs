use gl;
use gl::types::*;
use std::cell::RefCell;
use std::mem;
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

use crate::backend::gl::depth_test::depth_comparison_to_glenum;
use crate::backend::gl::pixel::opengl_pixel_format;
use crate::backend::gl::state::GLState;
use crate::backend::gl::GL;
use crate::backend::texture::{
  Dim, Dimensionable, GenMipmaps, Layerable, Layering, MagFilter, MinFilter, Sampler,
  Texture as TextureBackend, TextureBase, TextureError, Wrap,
};
use crate::pixel::{Pixel, PixelFormat};

pub struct Texture {
  pub(crate) handle: GLuint, // handle to the GPU texture object
  pub(crate) target: GLenum, // “type” of the texture; used for bindings
  mipmaps: usize,
  state: Rc<RefCell<GLState>>,
}

unsafe impl TextureBase for GL {
  type TextureRepr = Texture;
}

unsafe impl<L, D, P> TextureBackend<L, D, P> for GL
where
  D: Dimensionable,
  L: Layerable,
  P: Pixel,
{
  unsafe fn new_texture(
    &mut self,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self::TextureRepr, TextureError> {
    let mipmaps = mipmaps + 1; // + 1 prevent having 0 mipmaps
    let target = opengl_target(L::layering(), D::dim());

    let mut state = self.state.borrow_mut();

    let handle = state.generate_texture();
    state.bind_texture(target, handle);

    create_texture::<L, D>(target, size, mipmaps, P::pixel_format(), sampler)?;

    let texture = Texture {
      handle,
      target,
      mipmaps,
      state: self.state.clone(),
    };

    Ok(texture)
  }

  unsafe fn destroy_texture(texture: &mut Self::TextureRepr) {
    gl::DeleteTextures(1, &texture.handle);
  }

  unsafe fn mipmaps(texture: &Self::TextureRepr) -> usize {
    texture.mipmaps
  }

  unsafe fn clear_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<L, D, P>>::upload_part(
      texture,
      gen_mipmaps,
      offset,
      size,
      &vec![pixel; dim_capacity::<D>(size) as usize],
    )
  }

  unsafe fn clear(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<L, D, P>>::clear_part(texture, gen_mipmaps, D::ZERO_OFFSET, size, pixel)
  }

  unsafe fn upload_part(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    let mut gfx_state = texture.state.borrow_mut();

    gfx_state.bind_texture(texture.target, texture.handle);

    upload_texels::<L, D, P, P::Encoding>(texture.target, offset, size, texels)?;

    if gen_mipmaps == GenMipmaps::Yes {
      gl::GenerateMipmap(texture.target);
    }

    gfx_state.bind_texture(texture.target, 0);

    Ok(())
  }

  unsafe fn upload(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<L, D, P>>::upload_part(
      texture,
      gen_mipmaps,
      D::ZERO_OFFSET,
      size,
      texels,
    )
  }

  unsafe fn upload_part_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    let mut gfx_state = texture.state.borrow_mut();

    gfx_state.bind_texture(texture.target, texture.handle);

    upload_texels::<L, D, P, P::RawEncoding>(texture.target, offset, size, texels)?;

    if gen_mipmaps == GenMipmaps::Yes {
      gl::GenerateMipmap(texture.target);
    }

    gfx_state.bind_texture(texture.target, 0);

    Ok(())
  }

  unsafe fn upload_raw(
    texture: &mut Self::TextureRepr,
    gen_mipmaps: GenMipmaps,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    <Self as TextureBackend<L, D, P>>::upload_part_raw(
      texture,
      gen_mipmaps,
      D::ZERO_OFFSET,
      size,
      texels,
    )
  }

  unsafe fn get_raw_texels(
    texture: &Self::TextureRepr,
    _: D::Size,
  ) -> Result<Vec<P::RawEncoding>, TextureError>
  where
    P::RawEncoding: Copy + Default,
  {
    let mut texels = Vec::new();
    let pf = P::pixel_format();
    let (format, _, ty) = opengl_pixel_format(pf).unwrap();

    let mut w = 0;
    let mut h = 0;

    let mut gfx_state = texture.state.borrow_mut();
    gfx_state.bind_texture(texture.target, texture.handle);

    // retrieve the size of the texture (w and h)
    gl::GetTexLevelParameteriv(texture.target, 0, gl::TEXTURE_WIDTH, &mut w);
    gl::GetTexLevelParameteriv(texture.target, 0, gl::TEXTURE_HEIGHT, &mut h);

    // set the packing alignment based on the number of bytes to skip
    let skip_bytes = (pf.format.size() * w as usize) % 8;
    set_pack_alignment(skip_bytes);

    // resize the vec to allocate enough space to host the returned texels
    texels.resize_with((w * h) as usize * pf.canals_len(), Default::default);

    gl::GetTexImage(
      texture.target,
      0,
      format,
      ty,
      texels.as_mut_ptr() as *mut c_void,
    );

    gfx_state.bind_texture(texture.target, 0);

    Ok(texels)
  }
}

pub(crate) fn opengl_target(l: Layering, d: Dim) -> GLenum {
  match l {
    Layering::Flat => match d {
      Dim::Dim1 => gl::TEXTURE_1D,
      Dim::Dim2 => gl::TEXTURE_2D,
      Dim::Dim3 => gl::TEXTURE_3D,
      Dim::Cubemap => gl::TEXTURE_CUBE_MAP,
    },
    Layering::Layered => match d {
      Dim::Dim1 => gl::TEXTURE_1D_ARRAY,
      Dim::Dim2 => gl::TEXTURE_2D_ARRAY,
      Dim::Dim3 => unimplemented!("3D textures array not supported"),
      Dim::Cubemap => gl::TEXTURE_CUBE_MAP_ARRAY,
    },
  }
}

pub(crate) unsafe fn create_texture<L, D>(
  target: GLenum,
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
  sampler: Sampler,
) -> Result<(), TextureError>
where
  L: Layerable,
  D: Dimensionable,
{
  set_texture_levels(target, mipmaps);
  apply_sampler_to_texture(target, sampler);
  create_texture_storage::<L, D>(size, mipmaps, pf)
}

fn set_texture_levels(target: GLenum, mipmaps: usize) {
  unsafe {
    gl::TexParameteri(target, gl::TEXTURE_BASE_LEVEL, 0);
    gl::TexParameteri(target, gl::TEXTURE_MAX_LEVEL, mipmaps as GLint - 1);
  }
}

fn apply_sampler_to_texture(target: GLenum, sampler: Sampler) {
  unsafe {
    gl::TexParameteri(
      target,
      gl::TEXTURE_WRAP_R,
      opengl_wrap(sampler.wrap_r) as GLint,
    );
    gl::TexParameteri(
      target,
      gl::TEXTURE_WRAP_S,
      opengl_wrap(sampler.wrap_s) as GLint,
    );
    gl::TexParameteri(
      target,
      gl::TEXTURE_WRAP_T,
      opengl_wrap(sampler.wrap_t) as GLint,
    );
    gl::TexParameteri(
      target,
      gl::TEXTURE_MIN_FILTER,
      opengl_min_filter(sampler.min_filter) as GLint,
    );
    gl::TexParameteri(
      target,
      gl::TEXTURE_MAG_FILTER,
      opengl_mag_filter(sampler.mag_filter) as GLint,
    );

    match sampler.depth_comparison {
      Some(fun) => {
        gl::TexParameteri(
          target,
          gl::TEXTURE_COMPARE_FUNC,
          depth_comparison_to_glenum(fun) as GLint,
        );
        gl::TexParameteri(
          target,
          gl::TEXTURE_COMPARE_MODE,
          gl::COMPARE_REF_TO_TEXTURE as GLint,
        );
      }
      None => {
        gl::TexParameteri(target, gl::TEXTURE_COMPARE_MODE, gl::NONE as GLint);
      }
    }
  }
}

fn opengl_wrap(wrap: Wrap) -> GLenum {
  match wrap {
    Wrap::ClampToEdge => gl::CLAMP_TO_EDGE,
    Wrap::Repeat => gl::REPEAT,
    Wrap::MirroredRepeat => gl::MIRRORED_REPEAT,
  }
}

fn opengl_min_filter(filter: MinFilter) -> GLenum {
  match filter {
    MinFilter::Nearest => gl::NEAREST,
    MinFilter::Linear => gl::LINEAR,
    MinFilter::NearestMipmapNearest => gl::NEAREST_MIPMAP_NEAREST,
    MinFilter::NearestMipmapLinear => gl::NEAREST_MIPMAP_LINEAR,
    MinFilter::LinearMipmapNearest => gl::LINEAR_MIPMAP_NEAREST,
    MinFilter::LinearMipmapLinear => gl::LINEAR_MIPMAP_LINEAR,
  }
}

fn opengl_mag_filter(filter: MagFilter) -> GLenum {
  match filter {
    MagFilter::Nearest => gl::NEAREST,
    MagFilter::Linear => gl::LINEAR,
  }
}

fn create_texture_storage<L, D>(
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
) -> Result<(), TextureError>
where
  L: Layerable,
  D: Dimensionable,
{
  match opengl_pixel_format(pf) {
    Some(glf) => {
      let (format, iformat, encoding) = glf;

      match (L::layering(), D::dim()) {
        // 1D texture
        (Layering::Flat, Dim::Dim1) => {
          create_texture_1d_storage(format, iformat, encoding, D::width(size), mipmaps);
          Ok(())
        }

        // 2D texture
        (Layering::Flat, Dim::Dim2) => {
          create_texture_2d_storage(
            format,
            iformat,
            encoding,
            D::width(size),
            D::height(size),
            mipmaps,
          );
          Ok(())
        }

        // 3D texture
        (Layering::Flat, Dim::Dim3) => {
          create_texture_3d_storage(
            format,
            iformat,
            encoding,
            D::width(size),
            D::height(size),
            D::depth(size),
            mipmaps,
          );
          Ok(())
        }

        // cubemap
        (Layering::Flat, Dim::Cubemap) => {
          create_cubemap_storage(format, iformat, encoding, D::width(size), mipmaps);
          Ok(())
        }

        _ => Err(TextureError::TextureStorageCreationFailed(format!(
          "unsupported texture OpenGL pixel format: {:?}",
          glf
        ))),
      }
    }

    None => Err(TextureError::TextureStorageCreationFailed(format!(
      "unsupported texture pixel format: {:?}",
      pf
    ))),
  }
}

fn create_texture_1d_storage(
  format: GLenum,
  iformat: GLenum,
  encoding: GLenum,
  w: u32,
  mipmaps: usize,
) {
  for level in 0..mipmaps {
    let w = w / 2u32.pow(level as u32);

    unsafe {
      gl::TexImage1D(
        gl::TEXTURE_1D,
        level as GLint,
        iformat as GLint,
        w as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn create_texture_2d_storage(
  format: GLenum,
  iformat: GLenum,
  encoding: GLenum,
  w: u32,
  h: u32,
  mipmaps: usize,
) {
  for level in 0..mipmaps {
    let div = 2u32.pow(level as u32);
    let w = w / div;
    let h = h / div;

    unsafe {
      gl::TexImage2D(
        gl::TEXTURE_2D,
        level as GLint,
        iformat as GLint,
        w as GLsizei,
        h as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn create_texture_3d_storage(
  format: GLenum,
  iformat: GLenum,
  encoding: GLenum,
  w: u32,
  h: u32,
  d: u32,
  mipmaps: usize,
) {
  for level in 0..mipmaps {
    let div = 2u32.pow(level as u32);
    let w = w / div;
    let h = h / div;
    let d = d / div;

    unsafe {
      gl::TexImage3D(
        gl::TEXTURE_3D,
        level as GLint,
        iformat as GLint,
        w as GLsizei,
        h as GLsizei,
        d as GLsizei,
        0,
        format,
        encoding,
        ptr::null(),
      )
    };
  }
}

fn create_cubemap_storage(
  format: GLenum,
  iformat: GLenum,
  encoding: GLenum,
  s: u32,
  mipmaps: usize,
) {
  for level in 0..mipmaps {
    let s = s / 2u32.pow(level as u32);

    for face in 0..6 {
      unsafe {
        gl::TexImage2D(
          gl::TEXTURE_CUBE_MAP_POSITIVE_X + face,
          level as GLint,
          iformat as GLint,
          s as GLsizei,
          s as GLsizei,
          0,
          format,
          encoding,
          ptr::null(),
        )
      };
    }
  }
}

// set the unpack alignment for uploading aligned texels
fn set_unpack_alignment(skip_bytes: usize) {
  let unpack_alignment = match skip_bytes {
    0 => 8,
    2 => 2,
    4 => 4,
    _ => 1,
  };

  unsafe { gl::PixelStorei(gl::UNPACK_ALIGNMENT, unpack_alignment) };
}

// set the pack alignment for downloading aligned texels
fn set_pack_alignment(skip_bytes: usize) {
  let pack_alignment = match skip_bytes {
    0 => 8,
    2 => 2,
    4 => 4,
    _ => 1,
  };

  unsafe { gl::PixelStorei(gl::PACK_ALIGNMENT, pack_alignment) };
}

// Upload texels into the texture’s memory. Becareful of the type of texels you send down.
fn upload_texels<L, D, P, T>(
  target: GLenum,
  off: D::Offset,
  size: D::Size,
  texels: &[T],
) -> Result<(), TextureError>
where
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  // number of bytes in the input texels argument
  let input_bytes = texels.len() * mem::size_of::<T>();
  let pf = P::pixel_format();
  let pf_size = pf.format.size();
  let expected_bytes = D::count(size) * pf_size;

  if input_bytes < expected_bytes {
    // potential segfault / overflow; abort
    return Err(TextureError::NotEnoughPixels(expected_bytes, input_bytes));
  }

  // set the pixel row alignment to the required value for uploading data according to the width
  // of the texture and the size of a single pixel; here, skip_bytes represents the number of bytes
  // that will be skipped
  let skip_bytes = (D::width(size) as usize * pf_size) % 8;
  set_unpack_alignment(skip_bytes);

  match opengl_pixel_format(pf) {
    Some((format, _, encoding)) => match L::layering() {
      Layering::Flat => match D::dim() {
        Dim::Dim1 => unsafe {
          gl::TexSubImage1D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::width(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },

        Dim::Dim2 => unsafe {
          gl::TexSubImage2D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::y_offset(off) as GLint,
            D::width(size) as GLsizei,
            D::height(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },

        Dim::Dim3 => unsafe {
          gl::TexSubImage3D(
            target,
            0,
            D::x_offset(off) as GLint,
            D::y_offset(off) as GLint,
            D::z_offset(off) as GLint,
            D::width(size) as GLsizei,
            D::height(size) as GLsizei,
            D::depth(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },

        Dim::Cubemap => unsafe {
          gl::TexSubImage2D(
            gl::TEXTURE_CUBE_MAP_POSITIVE_X + D::z_offset(off),
            0,
            D::x_offset(off) as GLint,
            D::y_offset(off) as GLint,
            D::width(size) as GLsizei,
            D::width(size) as GLsizei,
            format,
            encoding,
            texels.as_ptr() as *const c_void,
          )
        },
      },

      Layering::Layered => unimplemented!("Layering::Layered not implemented yet"),
    },

    None => return Err(TextureError::UnsupportedPixelFormat(pf)),
  }

  Ok(())
}

// Capacity of the dimension, which is the product of the width, height and depth.
fn dim_capacity<D>(size: D::Size) -> u32
where
  D: Dimensionable,
{
  D::width(size) * D::height(size) * D::depth(size)
}
