//! Dynamic rendering pipelines.
//!
//! This module gives you types and functions to build *dynamic* rendering *pipelines*. A
//! pipeline represents a functional stream that consumes geometric data and rasterizes them.
//!
//! When you want to build a render, the main entry point is the `Builder` type. It enables you to
//! create dynamic `Pipeline` objects.
//!
//! # Key concepts
//!
//! luminance exposes several concepts you have to be familiar with:
//!
//!   - Render buffers.
//!   - Blending.
//!   - Shaders.
//!   - Tessellations.
//!   - Gates.
//!   - Render commands.
//!   - Shading commands.
//!   - Pipelines.
//!
//! # Render buffers
//!
//! The render buffers are GPU-allocated memory regions used while rendering images into
//! framebuffers. Typically, a framebuffer has at least three buffers:
//!
//!   - A *color buffer*, that will receive texels (akin to pixels, but for textures/buffers).
//!   - A *depth buffer*, a special buffer mostly used to determine whether a fragment (pixel) is behind
//!     something that was previously rendered – it’s a simple solution to discard render that won’t be
//!     visible anyway.
//!   - A *stencil buffer*, which often acts as a mask to create interesting effects to your renders.
//!
//! luminance gives you access to the first two – the stencil buffer will be added in a future
//! release.
//!
//! Alternatively, you can also tell your GPU that you won’t be using a depth buffer, or that you
//! need several color buffers – this is called [MRT](https://en.wikipedia.org/wiki/Multiple_Render_Targets).
//! In most frameworks, you create some textures to hold the color information, then you “bind” them
//! to your pipeline, and you’re good to go. In luminance, everything happens at the type level: you
//! say to luminance which type of framebuffer you want (for instance, a framebuffer with two color
//! outputs and a depth buffer). luminance will handle the textures for you – you can retrieve access
//! to them whenever you want to.
//!
//! If you have a depth buffer, you can ask luminance to perform a depth test that will discard any
//! fragment being “behind” the fragment already in place. You can also give luminance the *clear
//! color* it must use when you issue a new pipeline to fill the buffers.
//!
//! # Blending
//!
//! When you render a fragment A at a position P in a framebuffer, there are several configurations:
//!
//!   - You have a depth buffer and the depth test is enabled: in that case, no blending will happen as either
//!     the already in-place fragment will be chosen or the new one you try to write, depending on the result
//!     of the depth test.
//!   - The depth test is disabled: in that case, each time a fragment is to be written to a place in a
//!     buffer, its output will be blended with the color already present according to a *blending equation*
//!     and two *blending factors*.
//!
//! # Shaders
//!
//! Shaders in luminance are pretty simple: you have `Stage` that represents each type of shader
//! stages you can use, and `Program`, that links them into a GPU executable.
//!
//! In luminance, you’re supposed to create a `Program` with `Stage`. Some of them are mandatory and
//! others are optional. In order to customize your build, you can use a *uniform interface*. The
//! uniform interface is defined by our own type. When you create a program, you pass code to tell
//! luminance how to get such a type. Then that type will be handed back to you when needed in the
//! form of an immutable object.
//!
//! # Tessellations
//!
//! The `Tess` type represents a tessellation associated with a possible set of GPU vertices and
//! indices. Note that it’s also possible to create *attributeless* tesselations – i.e.
//! tessellations that don’t own any vertices nor indices and are used with specific shaders only.
//!
//! # Gates
//!
//! The gate concept is quite easy to understand: picture a pipeline as tree. Each node is typed,
//! hence, the structure is quite limited to what your GPU can understand. Gates are a way to
//! spread information of root nodes down. For instance, if you have a shader node, every nested
//! child of that shader node will have the possibility to use its features via a shading gate.
//!
//! # Render commands
//!
//! A set of values that tag a collection of tessellations. Typical information is whether we should
//! use a depth test, the blending factors and equation, etc.
//!
//! # Shading commands
//!
//! A set of values that tag a collection of render commands. Typical information is a shader
//! program along with its uniform interface.
//!
//! # Pipelines
//!
//! A pipeline is just an aggregation of shadings commands with a few extra information. It
//! especially gives you the power to scope-bind GPU resources.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

use crate::blending::BlendingState;
use crate::buffer::{Buffer, RawBuffer};
use crate::context::GraphicsContext;
use crate::depth_test::DepthTest;
use crate::face_culling::FaceCullingState;
use crate::framebuffer::{ColorSlot, DepthSlot, Framebuffer};
use crate::metagl::*;
use crate::pixel::{Pixel, SamplerType, Type as PxType};
use crate::render_state::RenderState;
use crate::shader::program::{
  Program, ProgramInterface, Type, Uniform, UniformInterface, Uniformable,
};
use crate::state::GraphicsState;
use crate::tess::TessSlice;
use crate::texture::{Dim, Dimensionable, Texture};
use crate::vertex::Semantics;

// A stack of bindings.
//
// This type implements a stacking system for effective resource bindings by allocating new
// bindings points only when no recycled resource is available. It helps have a better memory
// footprint in the resource space.
struct BindingStack {
  state: Rc<RefCell<GraphicsState>>,
  next_texture_unit: u32,
  free_texture_units: Vec<u32>,
  next_buffer_binding: u32,
  free_buffer_bindings: Vec<u32>,
}

impl BindingStack {
  // Create a new, empty binding stack.
  fn new(state: Rc<RefCell<GraphicsState>>) -> Self {
    BindingStack {
      state,
      next_texture_unit: 0,
      free_texture_units: Vec::new(),
      next_buffer_binding: 0,
      free_buffer_bindings: Vec::new(),
    }
  }
}

/// An opaque type used to create pipelines.
pub struct Builder<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
  binding_stack: Rc<RefCell<BindingStack>>,
  _borrow: PhantomData<&'a mut ()>,
}

impl<'a, C> Builder<'a, C>
where
  C: ?Sized + GraphicsContext,
{
  /// Create a new `Builder`.
  ///
  /// Even though you call this function by yourself, you’re likely to prefer using
  /// `GraphicsContext::pipeline_builder` instead.
  pub fn new(ctx: &'a mut C) -> Self {
    let state = ctx.state().clone();

    Builder {
      ctx,
      binding_stack: Rc::new(RefCell::new(BindingStack::new(state))),
      _borrow: PhantomData,
    }
  }

  /// Create a new [`Pipeline`] and consume it immediately.
  ///
  /// A dynamic rendering pipeline is responsible of rendering into a `Framebuffer`.
  ///
  /// `D` refers to the `Dim` of the underlying `Framebuffer`.
  ///
  /// `CS` and `DS` are – respectively – the *color* and *depth* `Slot`(s) of the underlying
  /// `Framebuffer`.
  ///
  /// Pipelines also have a *clear color*, used to clear the framebuffer.
  pub fn pipeline<'b, D, CS, DS, F>(
    &'b mut self,
    framebuffer: &Framebuffer<D, CS, DS>,
    pipeline_state: &PipelineState,
    f: F,
  ) where
    D: Dimensionable,
    CS: ColorSlot<D>,
    DS: DepthSlot<D>,
    F: FnOnce(Pipeline<'b>, ShadingGate<'b, C>),
  {
    unsafe {
      let mut state = self.ctx.state().borrow_mut();

      state.bind_draw_framebuffer(framebuffer.handle());

      let PipelineState {
        clear_color,
        clear_color_enabled,
        clear_depth_enabled,
        viewport,
        srgb_enabled,
      } = *pipeline_state;

      match viewport {
        Viewport::Whole => {
          state.set_viewport([
            0,
            0,
            framebuffer.width() as GLint,
            framebuffer.height() as GLint,
          ]);
        }

        Viewport::Specific {
          x,
          y,
          width,
          height,
        } => {
          state.set_viewport([x as GLint, y as GLint, width as GLint, height as GLint]);
        }
      }

      state.set_clear_color([
        clear_color[0] as _,
        clear_color[1] as _,
        clear_color[2] as _,
        clear_color[3] as _,
      ]);

      if clear_color_enabled || clear_depth_enabled {
        let color_bit = if clear_color_enabled {
          gl::COLOR_BUFFER_BIT
        } else {
          0
        };
        let depth_bit = if clear_depth_enabled {
          gl::DEPTH_BUFFER_BIT
        } else {
          0
        };
        gl::Clear(color_bit | depth_bit);
      }

      state.enable_srgb_framebuffer(srgb_enabled);
    }

    let binding_stack = &self.binding_stack;
    let p = Pipeline { binding_stack };
    let shd_gt = ShadingGate {
      ctx: self.ctx,
      binding_stack,
    };

    f(p, shd_gt);
  }
}

/// The viewport being part of the [`PipelineState`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Viewport {
  /// The whole viewport is used. The position and dimension of the viewport rectangle are
  /// extracted from the framebuffer.
  Whole,
  /// The viewport is specific and the rectangle area is user-defined.
  Specific {
    /// The lower position on the X axis to start the viewport rectangle at.
    x: u32,
    /// The lower position on the Y axis to start the viewport rectangle at.
    y: u32,
    /// The width of the viewport.
    width: u32,
    /// The height of the viewport.
    height: u32,
  },
}

/// Various customization options for pipelines.
#[derive(Clone, Debug)]
pub struct PipelineState {
  clear_color: [f32; 4],
  clear_color_enabled: bool,
  clear_depth_enabled: bool,
  viewport: Viewport,
  srgb_enabled: bool,
}

impl Default for PipelineState {
  /// Default [`PipelineState`]:
  ///
  /// - Clear color: `[0, 0, 0, 1]`.
  /// - Color is always cleared.
  /// - Depth is always cleared.
  /// - The viewport uses the whole framebuffer’s.
  /// - sRGB encoding is disabled.
  fn default() -> Self {
    PipelineState {
      clear_color: [0., 0., 0., 1.],
      clear_color_enabled: true,
      clear_depth_enabled: true,
      viewport: Viewport::Whole,
      srgb_enabled: false,
    }
  }
}

impl PipelineState {
  /// Create a default [`PipelineState`].
  ///
  /// See the documentation of the [`Default`] for further details.
  pub fn new() -> Self {
    Self::default()
  }

  /// Get the clear color.
  pub fn clear_color(&self) -> [f32; 4] {
    self.clear_color
  }

  /// Set the clear color.
  pub fn set_clear_color(self, clear_color: [f32; 4]) -> Self {
    Self {
      clear_color,
      ..self
    }
  }

  /// Check whether the pipeline’s framebuffer’s color buffers will be cleared.
  pub fn is_clear_color_enabled(&self) -> bool {
    self.clear_color_enabled
  }

  /// Enable clearing color buffers.
  pub fn enable_clear_color(self, clear_color_enabled: bool) -> Self {
    Self {
      clear_color_enabled,
      ..self
    }
  }

  /// Check whether the pipeline’s framebuffer’s depth buffer will be cleared.
  pub fn is_clear_depth_enabled(&self) -> bool {
    self.clear_depth_enabled
  }

  /// Enable clearing depth buffers.
  pub fn enable_clear_depth(self, clear_depth_enabled: bool) -> Self {
    Self {
      clear_depth_enabled,
      ..self
    }
  }

  /// Get the viewport.
  pub fn viewport(&self) -> Viewport {
    self.viewport
  }

  /// Set the viewport.
  pub fn set_viewport(self, viewport: Viewport) -> Self {
    Self { viewport, ..self }
  }

  /// Check whether sRGB linearization is enabled.
  pub fn is_srgb_enabled(&self) -> bool {
    self.srgb_enabled
  }

  /// Enable sRGB linearization.
  pub fn enable_srgb(self, srgb_enabled: bool) -> Self {
    Self {
      srgb_enabled,
      ..self
    }
  }
}

/// A dynamic pipeline.
///
/// Such a pipeline enables you to call shading commands, bind textures, bind uniform buffers, etc.
/// in a scoped-binding way.
pub struct Pipeline<'a> {
  binding_stack: &'a Rc<RefCell<BindingStack>>,
}

impl<'a> Pipeline<'a> {
  /// Bind a texture and return the bound texture.
  ///
  /// The texture remains bound as long as the return value lives.
  pub fn bind_texture<D, P>(
    &'a self,
    texture: &'a Texture<D, P>,
  ) -> BoundTexture<'a, D, P::SamplerType>
  where
    D: 'a + Dimensionable,
    P: 'a + Pixel,
  {
    let mut bstack = self.binding_stack.borrow_mut();

    let unit = bstack.free_texture_units.pop().unwrap_or_else(|| {
      // no more free units; reserve one
      let unit = bstack.next_texture_unit;
      bstack.next_texture_unit += 1;
      unit
    });

    unsafe {
      let mut state = bstack.state.borrow_mut();
      state.set_texture_unit(unit);
      state.bind_texture(texture.target(), texture.handle());
    }

    BoundTexture::new(self.binding_stack, unit)
  }

  /// Bind a buffer and return the bound buffer.
  ///
  /// The buffer remains bound as long as the return value lives.
  pub fn bind_buffer<T>(&'a self, buffer: &'a T) -> BoundBuffer<'a, T>
  where
    T: Deref<Target = RawBuffer>,
  {
    let mut bstack = self.binding_stack.borrow_mut();

    let binding = bstack.free_buffer_bindings.pop().unwrap_or_else(|| {
      // no more free bindings; reserve one
      let binding = bstack.next_buffer_binding;
      bstack.next_buffer_binding += 1;
      binding
    });

    unsafe {
      bstack
        .state
        .borrow_mut()
        .bind_buffer_base(buffer.handle(), binding);
    }

    BoundBuffer::new(self.binding_stack, binding)
  }
}

/// An opaque type representing a bound texture in a `Builder`. You may want to pass such an object
/// to a shader’s uniform’s update.
pub struct BoundTexture<'a, D, S>
where
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  unit: u32,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
  _t: PhantomData<&'a (D, S)>,
}

impl<'a, D, S> BoundTexture<'a, D, S>
where
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  fn new(binding_stack: &'a Rc<RefCell<BindingStack>>, unit: u32) -> Self {
    BoundTexture {
      unit,
      binding_stack,
      _t: PhantomData,
    }
  }
}

impl<'a, D, S> Drop for BoundTexture<'a, D, S>
where
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  fn drop(&mut self) {
    let mut bstack = self.binding_stack.borrow_mut();
    // place the unit into the free list
    bstack.free_texture_units.push(self.unit);
  }
}

unsafe impl<'a, 'b, D, S> Uniformable for &'b BoundTexture<'a, D, S>
where
  D: 'a + Dimensionable,
  S: 'a + SamplerType,
{
  fn update(self, u: &Uniform<Self>) {
    unsafe { gl::Uniform1i(u.index(), self.unit as GLint) }
  }

  fn ty() -> Type {
    match (S::sample_type(), D::dim()) {
      (PxType::NormIntegral, Dim::Dim1) => Type::Sampler1D,
      (PxType::NormUnsigned, Dim::Dim1) => Type::Sampler1D,
      (PxType::Integral, Dim::Dim1) => Type::ISampler1D,
      (PxType::Unsigned, Dim::Dim1) => Type::UISampler1D,
      (PxType::Floating, Dim::Dim1) => Type::Sampler1D,

      (PxType::NormIntegral, Dim::Dim2) => Type::Sampler2D,
      (PxType::NormUnsigned, Dim::Dim2) => Type::Sampler2D,
      (PxType::Integral, Dim::Dim2) => Type::ISampler2D,
      (PxType::Unsigned, Dim::Dim2) => Type::UISampler2D,
      (PxType::Floating, Dim::Dim2) => Type::Sampler2D,

      (PxType::NormIntegral, Dim::Dim3) => Type::Sampler3D,
      (PxType::NormUnsigned, Dim::Dim3) => Type::Sampler3D,
      (PxType::Integral, Dim::Dim3) => Type::ISampler3D,
      (PxType::Unsigned, Dim::Dim3) => Type::UISampler3D,
      (PxType::Floating, Dim::Dim3) => Type::Sampler3D,

      (PxType::NormIntegral, Dim::Cubemap) => Type::Cubemap,
      (PxType::NormUnsigned, Dim::Cubemap) => Type::Cubemap,
      (PxType::Integral, Dim::Cubemap) => Type::ICubemap,
      (PxType::Unsigned, Dim::Cubemap) => Type::UICubemap,
      (PxType::Floating, Dim::Cubemap) => Type::Cubemap,

      (PxType::NormIntegral, Dim::Dim1Array) => Type::Sampler1DArray,
      (PxType::NormUnsigned, Dim::Dim1Array) => Type::Sampler1DArray,
      (PxType::Integral, Dim::Dim1Array) => Type::ISampler1DArray,
      (PxType::Unsigned, Dim::Dim1Array) => Type::UISampler1DArray,
      (PxType::Floating, Dim::Dim1Array) => Type::Sampler1DArray,

      (PxType::NormIntegral, Dim::Dim2Array) => Type::Sampler2DArray,
      (PxType::NormUnsigned, Dim::Dim2Array) => Type::Sampler2DArray,
      (PxType::Integral, Dim::Dim2Array) => Type::ISampler2DArray,
      (PxType::Unsigned, Dim::Dim2Array) => Type::UISampler2DArray,
      (PxType::Floating, Dim::Dim2Array) => Type::Sampler2DArray,
    }
  }
}

/// An opaque type representing a bound buffer in a `Builder`. You may want to pass such an object
/// to a shader’s uniform’s update.
pub struct BoundBuffer<'a, T>
where
  T: 'a,
{
  binding: u32,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
  _t: PhantomData<&'a Buffer<T>>,
}

impl<'a, T> BoundBuffer<'a, T> {
  fn new(binding_stack: &'a Rc<RefCell<BindingStack>>, binding: u32) -> Self {
    BoundBuffer {
      binding,
      binding_stack,
      _t: PhantomData,
    }
  }
}

impl<'a, T> Drop for BoundBuffer<'a, T> {
  fn drop(&mut self) {
    let mut bstack = self.binding_stack.borrow_mut();
    // place the binding into the free list
    bstack.free_buffer_bindings.push(self.binding);
  }
}

unsafe impl<'a, 'b, T> Uniformable for &'b BoundBuffer<'a, T> {
  fn update(self, u: &Uniform<Self>) {
    unsafe { gl::UniformBlockBinding(u.program(), u.index() as GLuint, self.binding as GLuint) }
  }

  fn ty() -> Type {
    Type::BufferBinding
  }
}

/// A shading gate provides you with a way to run shaders on rendering commands.
pub struct ShadingGate<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
}

impl<'a, C> ShadingGate<'a, C>
where
  C: ?Sized + GraphicsContext,
{
  /// Run a shader on a set of rendering commands.
  pub fn shade<'b, In, Out, Uni, F>(&'b mut self, program: &Program<In, Out, Uni>, f: F)
  where
    In: Semantics,
    Uni: UniformInterface,
    F: FnOnce(ProgramInterface<Uni>, RenderGate<'b, C>),
  {
    unsafe {
      let bstack = self.binding_stack.borrow_mut();
      bstack.state.borrow_mut().use_program(program.handle());
    };

    let render_gate = RenderGate {
      ctx: self.ctx,
      binding_stack: self.binding_stack,
    };

    let program_interface = program.interface();
    f(program_interface, render_gate);
  }
}

/// Render gate, allowing you to alter the render state and render tessellations.
pub struct RenderGate<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
  binding_stack: &'a Rc<RefCell<BindingStack>>,
}

impl<'a, C> RenderGate<'a, C>
where
  C: ?Sized + GraphicsContext,
{
  /// Alter the render state and draw tessellations.
  pub fn render<'b, F>(&'b mut self, rdr_st: &RenderState, f: F)
  where
    F: FnOnce(TessGate<'b, C>),
  {
    unsafe {
      let bstack = self.binding_stack.borrow_mut();
      let mut gfx_state = bstack.state.borrow_mut();

      match rdr_st.blending {
        Some((equation, src_factor, dst_factor)) => {
          gfx_state.set_blending_state(BlendingState::On);
          gfx_state.set_blending_equation(equation);
          gfx_state.set_blending_func(src_factor, dst_factor);
        }
        None => {
          gfx_state.set_blending_state(BlendingState::Off);
        }
      }

      if let Some(depth_comparison) = rdr_st.depth_test {
        gfx_state.set_depth_test(DepthTest::On);
        gfx_state.set_depth_test_comparison(depth_comparison);
      } else {
        gfx_state.set_depth_test(DepthTest::Off);
      }

      match rdr_st.face_culling {
        Some(face_culling) => {
          gfx_state.set_face_culling_state(FaceCullingState::On);
          gfx_state.set_face_culling_order(face_culling.order);
          gfx_state.set_face_culling_mode(face_culling.mode);
        }
        None => {
          gfx_state.set_face_culling_state(FaceCullingState::Off);
        }
      }
    }

    let tess_gate = TessGate { ctx: self.ctx };

    f(tess_gate);
  }
}

/// Render tessellations.
pub struct TessGate<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
}

impl<'a, C> TessGate<'a, C>
where
  C: ?Sized + GraphicsContext,
{
  /// Render a tessellation.
  pub fn render<'b, T>(&'b mut self, tess: T)
  where
    T: Into<TessSlice<'b>>,
  {
    tess.into().render(self.ctx);
  }
}
