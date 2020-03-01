//! This module provides texture features.
//!
//! # Introduction to textures
//!
//! Textures are used intensively in graphics programs as they tend to be the *de facto* memory area
//! to store data. You use them typically when you want to customize a render, hold a render’s
//! texels or even store arbritrary data.
//!
//! Currently, the following textures are supported:
//!
//! - 1D, 2D and 3D textures
//! - cubemaps
//! - array of textures (any of the types above)
//!
//! Those combinations are encoded by several types. First of all, `Texture<D, P>` is the
//! polymorphic type used to represent textures. The `D` type variable is the dimension of the
//! texture. It can either be `Dim1`, `Dim2`, `Dim3`, `Cubemap`, `Dim1Array` or `Dim2Array`.
//! Finally, the `P` type variable is the pixel format the texture follows. See the `pixel` module
//! for further details about pixel formats.
//!
//! Additionally, all textures have between 0 or several *mipmaps*. Mipmaps are additional layers of
//! texels used to perform trilinear filtering in most applications. Those are low-definition images
//! of the the base image used to smoothly interpolate texels when a projection kicks in. See
//! [this](https://en.wikipedia.org/wiki/Mipmap) for more insight.
//!
//! # Creating textures
//!
//! Textures are created by providing a size, the number of mipmaps that should be used and a
//! reference to a `Sampler` object. Up to now, textures and samplers form the same object – but
//! that might change in the future. Samplers are just a way to describe how texels will be fetched
//! from a shader.
//!
//! ## Associated types
//!
//! Because textures might have different shapes, the types of their sizes and offsets vary. You
//! have to look at the implementation of `Dimensionable::Size` and `Dimensionable::Offset` to know
//! which type you have to pass. For instance, for a 2D texture – e.g. `Texture<Dim2, _>`, you have
//! to pass a pair `(width, height)`.
//!
//! ## Samplers
//!
//! Samplers gather filters – i.e. how a shader should interpolate texels while fetching them,
//! wrap rules – i.e. how a shader should behave when leaving the normalized UV coordinates? and
//! a depth comparison, for depth textures only. See the documentation of `Sampler` for further
//! explanations.
//!
//! Samplers must be declared in the shader code according to the type of the texture used in the
//! Rust code. The size won’t matter, only the type.
//!
//! # Uploading data to textures
//!
//! One of the primary use of textures is to store images so that they can be used in your
//! application mapped on objects in your scene, for instance. In order to do so, you have to load
//! the image from the disk – see the awesome [image](https://crates.io/crates/image) – and then
//! upload the data to the texture. You have several functions to do so:
//!
//! - `Texture::upload`: this function takes a slice of texels and upload them to the whole texture memory
//! - `Texture::upload_part`: this function does the same thing as `Texture::upload`, but gives you the extra
//!   control on where in the texture you want to upload and with which size
//! - `Texture::upload_raw`: this function takes a slice of raw encoding data and upload them to the whole
//!   texture memory. This is especially handy when your texture has several channels but the data you have
//!   don’t take channels into account and are just *raw* data.
//! - `Texture::upload_part_raw`: same thing as above, but with offset and size control.
//!
//! Alternatively, you can clear the texture with `Texture::clear` and `Texture::clear_part`.
//!
//! # Retrieving texels
//!
//! The function `Texel::get_raw_texels` must be used to retreive texels out of a texture. This
//! function allocates memory, so be careful when using it.
//!
//! [`PixelFormat`]: crate::pixel::PixelFormat

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;

use crate::context::GraphicsContext;
pub use crate::depth_test::DepthComparison;
use crate::metagl::*;
use crate::pixel::{opengl_pixel_format, Pixel, PixelFormat};
use crate::state::GraphicsState;

/// How to wrap texture coordinates while sampling textures?
#[derive(Clone, Copy, Debug)]
pub enum Wrap {
  /// If textures coordinates lay outside of *[0;1]*, they will be clamped to either *0* or *1* for
  /// every components.
  ClampToEdge,
  /// Textures coordinates are repeated if they lay outside of *[0;1]*. Picture this as:
  ///
  /// ```ignore
  /// // given the frac function returning the fractional part of a floating number:
  /// coord_ith = frac(coord_ith); // always between [0;1]
  /// ```
  Repeat,
  /// Same as `Repeat` but it will alternatively repeat between *[0;1]* and *[1;0]*.
  MirroredRepeat,
}

/// Minification filter.
#[derive(Clone, Copy, Debug)]
pub enum MinFilter {
  /// Nearest interpolation.
  Nearest,
  /// Linear interpolation between surrounding pixels.
  Linear,
  /// This filter will select the nearest mipmap between two samples and will perform a nearest
  /// interpolation afterwards.
  NearestMipmapNearest,
  /// This filter will select the nearest mipmap between two samples and will perform a linear
  /// interpolation afterwards.
  NearestMipmapLinear,
  /// This filter will linearly interpolate between two mipmaps, which selected texels would have
  /// been interpolated with a nearest filter.
  LinearMipmapNearest,
  /// This filter will linearly interpolate between two mipmaps, which selected texels would have
  /// been linarily interpolated as well.
  LinearMipmapLinear,
}

/// Magnification filter.
#[derive(Clone, Copy, Debug)]
pub enum MagFilter {
  /// Nearest interpolation.
  Nearest,
  /// Linear interpolation between surrounding pixels.
  Linear,
}

/// Reify a type into a `Dim`.
pub trait Dimensionable {
  /// Size type of a dimension (used to caracterize dimensions’ areas).
  type Size: Copy;

  /// Offset type of a dimension (used to caracterize addition and subtraction of sizes, mostly).
  type Offset: Copy;

  /// Zero offset.
  const ZERO_OFFSET: Self::Offset;

  /// Dimension.
  fn dim() -> Dim;

  /// Width of the associated `Size`.
  fn width(size: Self::Size) -> u32;

  /// Height of the associated `Size`. If it doesn’t have one, set it to 1.
  fn height(_: Self::Size) -> u32 {
    1
  }

  /// Depth of the associated `Size`. If it doesn’t have one, set it to 1.
  fn depth(_: Self::Size) -> u32 {
    1
  }

  /// X offset.
  fn x_offset(offset: Self::Offset) -> u32;

  /// Y offset. If it doesn’t have one, set it to 0.
  fn y_offset(_: Self::Offset) -> u32 {
    1
  }

  /// Z offset. If it doesn’t have one, set it to 0.
  fn z_offset(_: Self::Offset) -> u32 {
    1
  }

  /// Amount of pixels this size represents.
  ///
  /// For 2D sizes, it represents the area; for 3D sizes, the volume; etc.
  /// For cubemaps, it represents the side length of the cube.
  fn count(size: Self::Size) -> usize;
}

// Capacity of the dimension, which is the product of the width, height and depth.
fn dim_capacity<D>(size: D::Size) -> u32
where
  D: Dimensionable,
{
  D::width(size) * D::height(size) * D::depth(size)
}

/// Dimension of a texture.
#[derive(Clone, Copy, Debug)]
pub enum Dim {
  /// 1D.
  Dim1,
  /// 2D.
  Dim2,
  /// 3D.
  Dim3,
  /// Cubemap (i.e. a cube defining 6 faces — akin to 4D).
  Cubemap,
  /// 1D array.
  Dim1Array,
  /// 2D array.
  Dim2Array,
}

/// 1D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim1;

impl Dimensionable for Dim1 {
  type Offset = u32;
  type Size = u32;

  const ZERO_OFFSET: Self::Offset = 0;

  fn dim() -> Dim {
    Dim::Dim1
  }

  fn width(w: Self::Size) -> u32 {
    w
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off
  }

  fn count(size: Self::Size) -> usize {
    size as usize
  }
}

/// 2D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim2;

impl Dimensionable for Dim2 {
  type Offset = [u32; 2];
  type Size = [u32; 2];

  const ZERO_OFFSET: Self::Offset = [0, 0];

  fn dim() -> Dim {
    Dim::Dim2
  }

  fn width(size: Self::Size) -> u32 {
    size[0]
  }

  fn height(size: Self::Size) -> u32 {
    size[1]
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off[1]
  }

  fn count([width, height]: Self::Size) -> usize {
    width as usize * height as usize
  }
}

/// 3D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim3;

impl Dimensionable for Dim3 {
  type Offset = [u32; 3];
  type Size = [u32; 3];

  const ZERO_OFFSET: Self::Offset = [0, 0, 0];

  fn dim() -> Dim {
    Dim::Dim3
  }

  fn width(size: Self::Size) -> u32 {
    size[0]
  }

  fn height(size: Self::Size) -> u32 {
    size[1]
  }

  fn depth(size: Self::Size) -> u32 {
    size[2]
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off[1]
  }

  fn z_offset(off: Self::Offset) -> u32 {
    off[2]
  }

  fn count([width, height, depth]: Self::Size) -> usize {
    width as usize * height as usize * depth as usize
  }
}

/// Cubemap dimension.
#[derive(Clone, Copy, Debug)]
pub struct Cubemap;

impl Dimensionable for Cubemap {
  type Offset = ([u32; 2], CubeFace);
  type Size = u32;

  const ZERO_OFFSET: Self::Offset = ([0, 0], CubeFace::PositiveX);

  fn dim() -> Dim {
    Dim::Cubemap
  }

  fn width(s: Self::Size) -> u32 {
    s
  }

  fn height(s: Self::Size) -> u32 {
    s
  }

  fn depth(_: Self::Size) -> u32 {
    6
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off.0[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off.0[1]
  }

  fn z_offset(off: Self::Offset) -> u32 {
    match off.1 {
      CubeFace::PositiveX => 0,
      CubeFace::NegativeX => 1,
      CubeFace::PositiveY => 2,
      CubeFace::NegativeY => 3,
      CubeFace::PositiveZ => 4,
      CubeFace::NegativeZ => 5,
    }
  }

  fn count(size: Self::Size) -> usize {
    let size = size as usize;
    size * size
  }
}

/// Faces of a cubemap.
#[derive(Clone, Copy, Debug)]
pub enum CubeFace {
  /// The +X face of the cube.
  PositiveX,
  /// The -X face of the cube.
  NegativeX,
  /// The +Y face of the cube.
  PositiveY,
  /// The -Y face of the cube.
  NegativeY,
  /// The +Z face of the cube.
  PositiveZ,
  /// The -Z face of the cube.
  NegativeZ,
}

/// 1D array dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim1Array;

impl Dimensionable for Dim1Array {
  type Offset = (u32, u32);
  type Size = (u32, u32);

  const ZERO_OFFSET: Self::Offset = (0, 0);

  fn dim() -> Dim {
    Dim::Dim1Array
  }

  fn width(size: Self::Size) -> u32 {
    size.0
  }

  fn height(size: Self::Size) -> u32 {
    size.1
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off.0
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off.1
  }

  fn count((width, layer): Self::Size) -> usize {
    width as usize * layer as usize
  }
}

/// 3D dimension.
#[derive(Clone, Copy, Debug)]
pub struct Dim2Array;

impl Dimensionable for Dim2Array {
  type Offset = ([u32; 2], u32);
  type Size = ([u32; 2], u32);

  const ZERO_OFFSET: Self::Offset = ([0, 0], 0);

  fn dim() -> Dim {
    Dim::Dim2Array
  }

  fn width(size: Self::Size) -> u32 {
    size.0[0]
  }

  fn height(size: Self::Size) -> u32 {
    size.0[1]
  }

  fn depth(size: Self::Size) -> u32 {
    size.1
  }

  fn x_offset(off: Self::Offset) -> u32 {
    off.0[0]
  }

  fn y_offset(off: Self::Offset) -> u32 {
    off.0[1]
  }

  fn z_offset(off: Self::Offset) -> u32 {
    off.1
  }

  fn count(([width, height], layer): Self::Size) -> usize {
    width as usize * height as usize * layer as usize
  }
}

/// Raw buffer. Any buffer can be converted to that type. However, keep in mind that even though
/// type erasure is safe, creating a buffer from a raw buffer is not.
pub struct RawTexture {
  handle: GLuint, // handle to the GPU texture object
  target: GLenum, // “type” of the texture; used for bindings
  state: Rc<RefCell<GraphicsState>>,
}

impl RawTexture {
  pub(crate) unsafe fn new(
    state: Rc<RefCell<GraphicsState>>,
    handle: GLuint,
    target: GLenum,
  ) -> Self {
    RawTexture {
      handle,
      target,
      state,
    }
  }

  #[inline]
  pub(crate) fn handle(&self) -> GLuint {
    self.handle
  }

  #[inline]
  pub(crate) fn target(&self) -> GLenum {
    self.target
  }
}

/// Texture.
///
/// `D` refers to the dimension; `P` is the pixel format for the
/// texels.
pub struct Texture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  raw: RawTexture,
  size: D::Size,
  mipmaps: usize, // number of mipmaps
  _p: PhantomData<P>,
}

impl<D, P> Deref for Texture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  type Target = RawTexture;

  fn deref(&self) -> &Self::Target {
    &self.raw
  }
}

impl<D, P> DerefMut for Texture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.raw
  }
}

impl<D, P> Drop for Texture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  fn drop(&mut self) {
    unsafe { gl::DeleteTextures(1, &self.handle) }
  }
}

impl<D, P> Texture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  /// Create a new texture.
  ///
  ///   - The `mipmaps` parameter must be set to `0` if you want only one “layer of texels”.
  ///     creating a texture without any layer wouldn’t make any sense, so if you want three layers,
  ///     you will want the _base_ layer plus two mipmaps layers: you will then pass `2` as value
  ///     here.
  ///   - The `sampler` parameter allows to customize the way the texture will be sampled in
  ///     shader stages. Refer to the documentation of [`Sampler`] for further details.
  pub fn new<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    sampler: Sampler,
  ) -> Result<Self, TextureError>
  where
    C: GraphicsContext,
  {
    let mipmaps = mipmaps + 1; // + 1 prevent having 0 mipmaps
    let mut texture = 0;
    let target = opengl_target(D::dim());

    unsafe {
      gl::GenTextures(1, &mut texture);
      ctx.state().borrow_mut().bind_texture(target, texture);

      create_texture::<D>(target, size, mipmaps, P::pixel_format(), sampler)?;

      let raw = RawTexture::new(ctx.state().clone(), texture, target);

      Ok(Texture {
        raw,
        size,
        mipmaps,
        _p: PhantomData,
      })
    }
  }

  /// Create a texture from its backend representation.
  pub(crate) unsafe fn from_raw(raw: RawTexture, size: D::Size, mipmaps: usize) -> Self {
    Texture {
      raw,
      size,
      mipmaps: mipmaps + 1,
      _p: PhantomData,
    }
  }

  /// Convert a texture to its raw representation.
  pub fn into_raw(self) -> RawTexture {
    let raw = unsafe { ptr::read(&self.raw) };

    // forget self so that we don’t call drop on it after the function has returned
    mem::forget(self);
    raw
  }

  /// Number of mipmaps in the texture.
  #[inline(always)]
  pub fn mipmaps(&self) -> usize {
    self.mipmaps
  }

  /// Clear a part of a texture.
  ///
  /// The part being cleared is defined by a rectangle in which the `offset` represents the
  /// left-upper corner and the `size` gives the dimension of the rectangle. All the covered texels
  /// by this rectangle will be cleared to the `pixel` value.
  pub fn clear_part(
    &self,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    pixel: P::Encoding,
  ) -> Result<(), TextureError>
  where
    P::Encoding: Copy,
  {
    self.upload_part(
      gen_mipmaps,
      offset,
      size,
      &vec![pixel; dim_capacity::<D>(size) as usize],
    )
  }

  /// Clear a whole texture with a `pixel` value.
  pub fn clear(&self, gen_mipmaps: GenMipmaps, pixel: P::Encoding) -> Result<(), TextureError>
  where
    P::Encoding: Copy,
  {
    self.clear_part(gen_mipmaps, D::ZERO_OFFSET, self.size, pixel)
  }

  /// Upload texels to a part of a texture.
  ///
  /// The part being updated is defined by a rectangle in which the `offset` represents the
  /// left-upper corner and the `size` gives the dimension of the rectangle. All the covered texels
  /// by this rectangle will be updated by the `texels` slice.
  pub fn upload_part(
    &self,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    unsafe {
      let mut gfx_state = self.state.borrow_mut();

      gfx_state.bind_texture(self.target, self.handle);

      upload_texels::<D, P, P::Encoding>(self.target, offset, size, texels)?;

      if gen_mipmaps == GenMipmaps::Yes {
        gl::GenerateMipmap(self.target);
      }

      gfx_state.bind_texture(self.target, 0);
    }

    Ok(())
  }

  /// Upload `texels` to the whole texture.
  pub fn upload(
    &self,
    gen_mipmaps: GenMipmaps,
    texels: &[P::Encoding],
  ) -> Result<(), TextureError> {
    self.upload_part(gen_mipmaps, D::ZERO_OFFSET, self.size, texels)
  }

  /// Upload raw `texels` to a part of a texture.
  ///
  /// This function is similar to `upload_part` but it works on `P::RawEncoding` instead of
  /// `P::Encoding`. This useful when the texels are represented as a contiguous array of raw
  /// components of the texels.
  pub fn upload_part_raw(
    &self,
    gen_mipmaps: GenMipmaps,
    offset: D::Offset,
    size: D::Size,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    unsafe {
      let mut gfx_state = self.state.borrow_mut();

      gfx_state.bind_texture(self.target, self.handle);

      upload_texels::<D, P, P::RawEncoding>(self.target, offset, size, texels)?;

      if gen_mipmaps == GenMipmaps::Yes {
        gl::GenerateMipmap(self.target);
      }

      gfx_state.bind_texture(self.target, 0);
    }

    Ok(())
  }

  /// Upload raw `texels` to the whole texture.
  pub fn upload_raw(
    &self,
    gen_mipmaps: GenMipmaps,
    texels: &[P::RawEncoding],
  ) -> Result<(), TextureError> {
    self.upload_part_raw(gen_mipmaps, D::ZERO_OFFSET, self.size, texels)
  }

  // FIXME: cubemaps?
  /// Get the raw texels associated with this texture.
  pub fn get_raw_texels(&self) -> Vec<P::RawEncoding>
  where
    P: Pixel,
    P::RawEncoding: Copy + Default,
  {
    let mut texels = Vec::new();
    let pf = P::pixel_format();
    let (format, _, ty) = opengl_pixel_format(pf).unwrap();

    unsafe {
      let mut w = 0;
      let mut h = 0;

      let mut gfx_state = self.state.borrow_mut();
      gfx_state.bind_texture(self.target, self.handle);

      // retrieve the size of the texture (w and h)
      gl::GetTexLevelParameteriv(self.target, 0, gl::TEXTURE_WIDTH, &mut w);
      gl::GetTexLevelParameteriv(self.target, 0, gl::TEXTURE_HEIGHT, &mut h);

      // set the packing alignment based on the number of bytes to skip
      let skip_bytes = (pf.format.size() * w as usize) % 8;
      set_pack_alignment(skip_bytes);

      // resize the vec to allocate enough space to host the returned texels
      texels.resize_with((w * h) as usize * pf.canals_len(), Default::default);

      gl::GetTexImage(
        self.target,
        0,
        format,
        ty,
        texels.as_mut_ptr() as *mut c_void,
      );

      gfx_state.bind_texture(self.target, 0);
    }

    texels
  }

  /// Get the inner size of the texture.
  ///
  /// That value represents the _dimension_ of the texture. Depending on the type of texture, its
  /// dimensionality varies. For instance:
  ///
  ///   - 1D textures have a single value, giving the length of the texture.
  ///   - 2D textures have two values for their _width_ and _height_.
  ///   - 3D textures have three values: _width_, _height_ and _depth_.
  ///   - Etc. etc.
  pub fn size(&self) -> D::Size {
    self.size
  }
}

/// Whether mipmaps should be generated.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GenMipmaps {
  /// Mipmaps should be generated.
  ///
  /// Mipmaps are generated when creating textures but also when uploading texels, clearing, etc.
  Yes,
  /// Never generate mipmaps.
  No,
}

pub(crate) fn opengl_target(d: Dim) -> GLenum {
  match d {
    Dim::Dim1 => gl::TEXTURE_1D,
    Dim::Dim2 => gl::TEXTURE_2D,
    Dim::Dim3 => gl::TEXTURE_3D,
    Dim::Cubemap => gl::TEXTURE_CUBE_MAP,
    Dim::Dim1Array => gl::TEXTURE_1D_ARRAY,
    Dim::Dim2Array => gl::TEXTURE_2D_ARRAY,
  }
}

pub(crate) unsafe fn create_texture<D>(
  target: GLenum,
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
  sampler: Sampler,
) -> Result<(), TextureError>
where
  D: Dimensionable,
{
  set_texture_levels(target, mipmaps);
  apply_sampler_to_texture(target, sampler);
  create_texture_storage::<D>(size, mipmaps, pf)
}

fn create_texture_storage<D>(
  size: D::Size,
  mipmaps: usize,
  pf: PixelFormat,
) -> Result<(), TextureError>
where
  D: Dimensionable,
{
  match opengl_pixel_format(pf) {
    Some(glf) => {
      let (format, iformat, encoding) = glf;

      match D::dim() {
        // 1D texture
        Dim::Dim1 => {
          create_texture_1d_storage(format, iformat, encoding, D::width(size), mipmaps);
          Ok(())
        }

        // 2D texture
        Dim::Dim2 => {
          create_texture_2d_storage(
            gl::TEXTURE_2D,
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
        Dim::Dim3 => {
          create_texture_3d_storage(
            gl::TEXTURE_3D,
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
        Dim::Cubemap => {
          create_cubemap_storage(format, iformat, encoding, D::width(size), mipmaps);
          Ok(())
        }

        // 1D array texture
        Dim::Dim1Array => {
          create_texture_2d_storage(
            gl::TEXTURE_1D_ARRAY,
            format,
            iformat,
            encoding,
            D::width(size),
            D::height(size),
            mipmaps,
          );
          Ok(())
        }

        // 2D array texture
        Dim::Dim2Array => {
          create_texture_3d_storage(
            gl::TEXTURE_2D_ARRAY,
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
  target: GLenum,
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
        target,
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
  target: GLenum,
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
        target,
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
        gl::TexParameteri(target, gl::TEXTURE_COMPARE_FUNC, fun.to_glenum() as GLint);
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
fn upload_texels<D, P, T>(
  target: GLenum,
  off: D::Offset,
  size: D::Size,
  texels: &[T],
) -> Result<(), TextureError>
where
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
    Some((format, _, encoding)) => match D::dim() {
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

      Dim::Dim1Array => unsafe {
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

      Dim::Dim2Array => unsafe {
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
    },

    None => return Err(TextureError::UnsupportedPixelFormat(pf)),
  }

  Ok(())
}

/// A `Sampler` object gives hint on how a `Texture` should be sampled.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
  /// How should we wrap around the *r* sampling coordinate?
  pub wrap_r: Wrap,
  /// How should we wrap around the *s* sampling coordinate?
  pub wrap_s: Wrap,
  /// How should we wrap around the *t* sampling coordinate?
  pub wrap_t: Wrap,
  /// Minification filter.
  pub min_filter: MinFilter,
  /// Magnification filter.
  pub mag_filter: MagFilter,
  /// For depth textures, should we perform depth comparison and if so, how?
  pub depth_comparison: Option<DepthComparison>,
}

/// Default value is as following:
impl Default for Sampler {
  fn default() -> Self {
    Sampler {
      wrap_r: Wrap::ClampToEdge,
      wrap_s: Wrap::ClampToEdge,
      wrap_t: Wrap::ClampToEdge,
      min_filter: MinFilter::NearestMipmapLinear,
      mag_filter: MagFilter::Linear,
      depth_comparison: None,
    }
  }
}

/// Errors that might happen when working with textures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextureError {
  /// A texture’s storage failed to be created.
  ///
  /// The carried [`String`] gives the reason of the failure.
  TextureStorageCreationFailed(String),
  /// Not enough pixel data provided for the given area asked.
  ///
  /// The first [`usize`] is the number of expected bytes to be uploaded and the second [`usize`] is
  /// the number you provided. You must provide at least as many pixels as expected by the area in
  /// the texture you’re uploading to.
  NotEnoughPixels(usize, usize),
  /// Unsupported pixel format.
  ///
  /// Sometimes, some hardware might not support a given pixel format (or the format exists on
  /// the interface side but doesn’t in the implementation). That error represents such a case.
  UnsupportedPixelFormat(PixelFormat),
}

impl fmt::Display for TextureError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      TextureError::TextureStorageCreationFailed(ref e) => {
        write!(f, "texture storage creation failed: {}", e)
      }

      TextureError::NotEnoughPixels(expected, provided) => write!(
        f,
        "not enough texels provided: expected {} bytes, provided {} bytes",
        expected, provided
      ),

      TextureError::UnsupportedPixelFormat(fmt) => write!(f, "unsupported pixel format: {:?}", fmt),
    }
  }
}
