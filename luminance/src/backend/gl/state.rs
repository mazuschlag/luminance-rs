//! Graphics state.

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;

use crate::backend::gl::depth_test::depth_comparison_to_glenum;
use crate::blending::{BlendingState, Equation, Factor};
use crate::depth_test::{DepthComparison, DepthTest};
use crate::face_culling::{FaceCullingMode, FaceCullingOrder, FaceCullingState};
use crate::metagl::*;
use crate::vertex_restart::VertexRestart;

// TLS synchronization barrier for `GLState`.
//
// Note: disable on no_std.
thread_local!(static TLS_ACQUIRE_GFX_STATE: RefCell<Option<()>> = RefCell::new(Some(())));

pub struct BindingStack {
  next_texture_unit: u32,
  free_texture_units: Vec<u32>,
  next_buffer_binding: u32,
  free_buffer_bindings: Vec<u32>,
}

impl BindingStack {
  // Create a new, empty binding stack.
  fn new() -> Self {
    BindingStack {
      next_texture_unit: 0,
      free_texture_units: Vec::new(),
      next_buffer_binding: 0,
      free_buffer_bindings: Vec::new(),
    }
  }
}

/// The graphics state.
///
/// This type represents the current state of a given graphics context. It acts
/// as a forward-gate to all the exposed features from the low-level API but
/// adds a small cache layer over it to prevent from issuing the same API call (with
/// the same parameters).
pub struct GLState {
  _a: PhantomData<*const ()>, // !Send and !Sync

  // binding stack
  binding_stack: BindingStack,

  // viewport
  viewport: [GLint; 4],

  // clear buffers
  clear_color: [GLfloat; 4],

  // blending
  blending_state: BlendingState,
  blending_equation: Equation,
  blending_func: (Factor, Factor),

  // depth test
  depth_test: DepthTest,
  depth_test_comparison: DepthComparison,

  // face culling
  face_culling_state: FaceCullingState,
  face_culling_order: FaceCullingOrder,
  face_culling_mode: FaceCullingMode,

  // vertex restart
  vertex_restart: VertexRestart,

  // patch primitive vertex number
  patch_vertex_nb: usize,

  // texture
  current_texture_unit: GLenum,
  bound_textures: Vec<(GLenum, GLuint)>,

  // texture buffer used to optimize texture creation; regular textures typically will never ask
  // for fetching from this set but framebuffers, who often generate several textures, might use
  // this opportunity to get N textures (color, depth and stencil) at once, in a single CPU / GPU
  // roundtrip
  //
  // fishy fishy
  texture_swimming_pool: Vec<GLuint>,

  // uniform buffer
  bound_uniform_buffers: Vec<GLuint>,

  // array buffer
  bound_array_buffer: GLuint,

  // element buffer
  bound_element_array_buffer: GLuint,

  // framebuffer
  bound_draw_framebuffer: GLuint,

  // vertex array
  bound_vertex_array: GLuint,

  // shader program
  current_program: GLuint,

  // framebuffer sRGB
  srgb_framebuffer_enabled: bool,
}

impl GLState {
  /// Create a new `GLState`.
  ///
  /// > Note: keep in mind you can create only one per thread. However, if you’re building without
  /// > standard library, this function will always return successfully. You have to take extra care
  /// > in this case.
  pub fn new() -> Result<Self, StateQueryError> {
    TLS_ACQUIRE_GFX_STATE.with(|rc| {
      let mut inner = rc.borrow_mut();

      match *inner {
        Some(_) => {
          inner.take();
          Self::get_from_context()
        }

        None => Err(StateQueryError::UnavailableGLState),
      }
    })
  }

  /// Get a `GraphicsContext` from the current OpenGL context.
  pub(crate) fn get_from_context() -> Result<Self, StateQueryError> {
    unsafe {
      let binding_stack = BindingStack::new();
      let viewport = get_ctx_viewport()?;
      let clear_color = get_ctx_clear_color()?;
      let blending_state = get_ctx_blending_state()?;
      let blending_equation = get_ctx_blending_equation()?;
      let blending_func = get_ctx_blending_factors()?;
      let depth_test = get_ctx_depth_test()?;
      let depth_test_comparison = DepthComparison::Less;
      let face_culling_state = get_ctx_face_culling_state()?;
      let face_culling_order = get_ctx_face_culling_order()?;
      let face_culling_mode = get_ctx_face_culling_mode()?;
      let vertex_restart = get_ctx_vertex_restart()?;
      let patch_vertex_nb = 0;
      let current_texture_unit = get_ctx_current_texture_unit()?;
      let bound_textures = vec![(gl::TEXTURE_2D, 0); 48]; // 48 is the platform minimal requirement
      let texture_swimming_pool = Vec::new();
      let bound_uniform_buffers = vec![0; 36]; // 36 is the platform minimal requirement
      let bound_array_buffer = 0;
      let bound_element_array_buffer = 0;
      let bound_draw_framebuffer = get_ctx_bound_draw_framebuffer()?;
      let bound_vertex_array = get_ctx_bound_vertex_array()?;
      let current_program = get_ctx_current_program()?;
      let srgb_framebuffer_enabled = get_ctx_srgb_framebuffer_enabled()?;

      Ok(GLState {
        _a: PhantomData,
        binding_stack,
        viewport,
        clear_color,
        blending_state,
        blending_equation,
        blending_func,
        depth_test,
        depth_test_comparison,
        face_culling_state,
        face_culling_order,
        face_culling_mode,
        vertex_restart,
        patch_vertex_nb,
        current_texture_unit,
        bound_textures,
        texture_swimming_pool,
        bound_uniform_buffers,
        bound_array_buffer,
        bound_element_array_buffer,
        bound_draw_framebuffer,
        bound_vertex_array,
        current_program,
        srgb_framebuffer_enabled,
      })
    }
  }

  pub(crate) fn generate_texture(&mut self) -> GLuint {
    self.texture_swimming_pool.pop().unwrap_or_else(|| {
      let mut texture = 0;

      unsafe { gl::GenTextures(1, &mut texture) };
      texture
    })
  }

  /// Reserve at least a given number of textures.
  pub(crate) fn reserve_textures(&mut self, nb: usize) {
    let available = self.texture_swimming_pool.len();
    let needed = nb.max(available) - available;

    if needed > 0 {
      // resize the internal buffer to hold all the new textures and create a slice starting from
      // the previous end to the new end
      self.texture_swimming_pool.resize(available + needed, 0);
      let textures = &mut self.texture_swimming_pool[available..];

      unsafe { gl::GenTextures(1, textures.as_mut_ptr()) };
    }
  }

  pub(crate) unsafe fn set_viewport(&mut self, viewport: [GLint; 4]) {
    if self.viewport != viewport {
      gl::Viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
      self.viewport = viewport;
    }
  }

  pub(crate) unsafe fn set_clear_color(&mut self, clear_color: [GLfloat; 4]) {
    if self.clear_color != clear_color {
      gl::ClearColor(
        clear_color[0],
        clear_color[1],
        clear_color[2],
        clear_color[3],
      );
      self.clear_color = clear_color;
    }
  }

  pub(crate) unsafe fn set_blending_state(&mut self, state: BlendingState) {
    if self.blending_state != state {
      match state {
        BlendingState::On => gl::Enable(gl::BLEND),
        BlendingState::Off => gl::Disable(gl::BLEND),
      }

      self.blending_state = state;
    }
  }

  pub(crate) unsafe fn set_blending_equation(&mut self, equation: Equation) {
    if self.blending_equation != equation {
      gl::BlendEquation(from_blending_equation(equation));
      self.blending_equation = equation;
    }
  }

  pub(crate) unsafe fn set_blending_func(&mut self, src: Factor, dest: Factor) {
    if self.blending_func != (src, dest) {
      gl::BlendFunc(from_blending_factor(src), from_blending_factor(dest));
      self.blending_func = (src, dest);
    }
  }

  pub(crate) unsafe fn set_depth_test(&mut self, depth_test: DepthTest) {
    if self.depth_test != depth_test {
      match depth_test {
        DepthTest::On => gl::Enable(gl::DEPTH_TEST),
        DepthTest::Off => gl::Disable(gl::DEPTH_TEST),
      }

      self.depth_test = depth_test;
    }
  }

  pub(crate) unsafe fn set_depth_test_comparison(
    &mut self,
    depth_test_comparison: DepthComparison,
  ) {
    if self.depth_test_comparison != depth_test_comparison {
      gl::DepthFunc(depth_comparison_to_glenum(depth_test_comparison));
      self.depth_test_comparison = depth_test_comparison;
    }
  }

  pub(crate) unsafe fn set_face_culling_state(&mut self, state: FaceCullingState) {
    if self.face_culling_state != state {
      match state {
        FaceCullingState::On => gl::Enable(gl::CULL_FACE),
        FaceCullingState::Off => gl::Disable(gl::CULL_FACE),
      }

      self.face_culling_state = state;
    }
  }

  pub(crate) unsafe fn set_face_culling_order(&mut self, order: FaceCullingOrder) {
    if self.face_culling_order != order {
      match order {
        FaceCullingOrder::CW => gl::FrontFace(gl::CW),
        FaceCullingOrder::CCW => gl::FrontFace(gl::CCW),
      }

      self.face_culling_order = order;
    }
  }

  pub(crate) unsafe fn set_face_culling_mode(&mut self, mode: FaceCullingMode) {
    if self.face_culling_mode != mode {
      match mode {
        FaceCullingMode::Front => gl::CullFace(gl::FRONT),
        FaceCullingMode::Back => gl::CullFace(gl::BACK),
        FaceCullingMode::Both => gl::CullFace(gl::FRONT_AND_BACK),
      }

      self.face_culling_mode = mode;
    }
  }

  pub(crate) unsafe fn set_vertex_restart(&mut self, state: VertexRestart) {
    if self.vertex_restart != state {
      match state {
        VertexRestart::On => gl::Enable(gl::PRIMITIVE_RESTART),
        VertexRestart::Off => gl::Disable(gl::PRIMITIVE_RESTART),
      }
    }
  }

  pub(crate) unsafe fn set_patch_vertex_nb(&mut self, nb: usize) {
    if self.patch_vertex_nb != nb {
      gl::PatchParameteri(gl::PATCH_VERTICES, nb as GLint);
      self.patch_vertex_nb = nb;
    }
  }

  pub(crate) unsafe fn set_texture_unit(&mut self, unit: u32) {
    if self.current_texture_unit != unit {
      gl::ActiveTexture(gl::TEXTURE0 + unit as GLenum);
      self.current_texture_unit = unit;
    }
  }

  pub(crate) unsafe fn bind_texture(&mut self, target: GLenum, handle: GLuint) {
    let unit = self.current_texture_unit as usize;

    match self.bound_textures.get(unit).cloned() {
      Some((target_, handle_)) if target != target_ || handle != handle_ => {
        gl::BindTexture(target, handle);
        self.bound_textures[unit] = (target, handle);
      }

      None => {
        gl::BindTexture(target, handle);

        // not enough registered texture units; let’s grow a bit more
        self.bound_textures.resize(unit + 1, (gl::TEXTURE_2D, 0));
        self.bound_textures[unit] = (target, handle);
      }

      _ => (), // cached
    }
  }

  pub(crate) unsafe fn bind_buffer_base(&mut self, handle: GLuint, binding: u32) {
    let binding_ = binding as usize;

    match self.bound_uniform_buffers.get(binding_).cloned() {
      Some(handle_) if handle != handle_ => {
        gl::BindBufferBase(gl::UNIFORM_BUFFER, binding as GLuint, handle);
        self.bound_uniform_buffers[binding_] = handle;
      }

      None => {
        gl::BindBufferBase(gl::UNIFORM_BUFFER, binding as GLuint, handle);

        // not enough registered buffer bindings; let’s grow a bit more
        self.bound_uniform_buffers.resize(binding_ + 1, 0);
        self.bound_uniform_buffers[binding_] = handle;
      }

      _ => (), // cached
    }
  }

  pub(crate) unsafe fn bind_array_buffer(&mut self, handle: GLuint, bind: Bind) {
    if bind == Bind::Forced || self.bound_array_buffer != handle {
      gl::BindBuffer(gl::ARRAY_BUFFER, handle);
      self.bound_array_buffer = handle;
    }
  }

  pub(crate) unsafe fn bind_element_array_buffer(&mut self, handle: GLuint, bind: Bind) {
    if bind == Bind::Forced || self.bound_element_array_buffer != handle {
      gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, handle);
      self.bound_element_array_buffer = handle;
    }
  }

  pub(crate) unsafe fn unbind_buffer(&mut self, handle: GLuint) {
    if self.bound_array_buffer == handle {
      self.bind_array_buffer(0, Bind::Cached);
    } else if self.bound_element_array_buffer == handle {
      self.bind_element_array_buffer(0, Bind::Cached);
    } else if let Some(handle_) = self
      .bound_uniform_buffers
      .iter_mut()
      .find(|h| **h == handle)
    {
      *handle_ = 0;
    }
  }

  pub(crate) unsafe fn bind_draw_framebuffer(&mut self, handle: GLuint) {
    if self.bound_draw_framebuffer != handle {
      gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, handle);
      self.bound_draw_framebuffer = handle;
    }
  }

  pub(crate) unsafe fn bind_vertex_array(&mut self, handle: GLuint, bind: Bind) {
    if bind == Bind::Forced || self.bound_vertex_array != handle {
      gl::BindVertexArray(handle);
      self.bound_vertex_array = handle;
    }
  }

  pub(crate) unsafe fn unbind_vertex_array(&mut self) {
    self.bind_vertex_array(0, Bind::Cached)
  }

  pub(crate) unsafe fn use_program(&mut self, handle: GLuint) {
    if self.current_program != handle {
      gl::UseProgram(handle);
      self.current_program = handle;
    }
  }

  pub(crate) unsafe fn enable_srgb_framebuffer(&mut self, srgb_framebuffer_enabled: bool) {
    if self.srgb_framebuffer_enabled != srgb_framebuffer_enabled {
      if srgb_framebuffer_enabled {
        gl::Enable(gl::FRAMEBUFFER_SRGB);
      } else {
        gl::Disable(gl::FRAMEBUFFER_SRGB);
      }

      self.srgb_framebuffer_enabled = srgb_framebuffer_enabled;
    }
  }
}

/// Should the binding be cached or forced to the provided value?
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Bind {
  Forced,
  Cached,
}

#[inline]
fn from_blending_equation(equation: Equation) -> GLenum {
  match equation {
    Equation::Additive => gl::FUNC_ADD,
    Equation::Subtract => gl::FUNC_SUBTRACT,
    Equation::ReverseSubtract => gl::FUNC_REVERSE_SUBTRACT,
    Equation::Min => gl::MIN,
    Equation::Max => gl::MAX,
  }
}

#[inline]
fn from_blending_factor(factor: Factor) -> GLenum {
  match factor {
    Factor::One => gl::ONE,
    Factor::Zero => gl::ZERO,
    Factor::SrcColor => gl::SRC_COLOR,
    Factor::SrcColorComplement => gl::ONE_MINUS_SRC_COLOR,
    Factor::DestColor => gl::DST_COLOR,
    Factor::DestColorComplement => gl::ONE_MINUS_DST_COLOR,
    Factor::SrcAlpha => gl::SRC_ALPHA,
    Factor::SrcAlphaComplement => gl::ONE_MINUS_SRC_ALPHA,
    Factor::DstAlpha => gl::DST_ALPHA,
    Factor::DstAlphaComplement => gl::ONE_MINUS_DST_ALPHA,
    Factor::SrcAlphaSaturate => gl::SRC_ALPHA_SATURATE,
  }
}

/// An error that might happen when the context is queried.
#[derive(Debug)]
pub enum StateQueryError {
  /// The [`GLState`] object is unavailable.
  ///
  /// That might occur if the current thread doesn’t support allocating a new graphics state. It
  /// might happen if you try to have more than one state on the same thread, for instance.
  UnavailableGLState,
  /// Corrupted blending state.
  UnknownBlendingState(GLboolean),
  /// Corrupted blending equation.
  UnknownBlendingEquation(GLenum),
  /// Corrupted blending source factor.
  UnknownBlendingSrcFactor(GLenum),
  /// Corrupted blending destination factor.
  UnknownBlendingDstFactor(GLenum),
  /// Corrupted depth test state.
  UnknownDepthTestState(GLboolean),
  /// Corrupted face culling state.
  UnknownFaceCullingState(GLboolean),
  /// Corrupted face culling order.
  UnknownFaceCullingOrder(GLenum),
  /// Corrupted face culling mode.
  UnknownFaceCullingMode(GLenum),
  /// Corrupted vertex restart state.
  UnknownVertexRestartState(GLboolean),
  /// Corrupted sRGB framebuffer state.
  UnknownSRGBFramebufferState(GLboolean),
}

impl fmt::Display for StateQueryError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StateQueryError::UnavailableGLState => write!(f, "unavailable graphics state"),
      StateQueryError::UnknownBlendingState(ref s) => write!(f, "unknown blending state: {}", s),
      StateQueryError::UnknownBlendingEquation(ref e) => {
        write!(f, "unknown blending equation: {}", e)
      }
      StateQueryError::UnknownBlendingSrcFactor(ref k) => {
        write!(f, "unknown blending source factor: {}", k)
      }
      StateQueryError::UnknownBlendingDstFactor(ref k) => {
        write!(f, "unknown blending destination factor: {}", k)
      }
      StateQueryError::UnknownDepthTestState(ref s) => write!(f, "unknown depth test state: {}", s),
      StateQueryError::UnknownFaceCullingState(ref s) => {
        write!(f, "unknown face culling state: {}", s)
      }
      StateQueryError::UnknownFaceCullingOrder(ref o) => {
        write!(f, "unknown face culling order: {}", o)
      }
      StateQueryError::UnknownFaceCullingMode(ref m) => {
        write!(f, "unknown face culling mode: {}", m)
      }
      StateQueryError::UnknownVertexRestartState(ref s) => {
        write!(f, "unknown vertex restart state: {}", s)
      }
      StateQueryError::UnknownSRGBFramebufferState(ref s) => {
        write!(f, "unknown sRGB framebuffer state: {}", s)
      }
    }
  }
}

unsafe fn get_ctx_viewport() -> Result<[GLint; 4], StateQueryError> {
  let mut data = [0; 4];
  gl::GetIntegerv(gl::VIEWPORT, data.as_mut_ptr());
  Ok(data)
}

unsafe fn get_ctx_clear_color() -> Result<[GLfloat; 4], StateQueryError> {
  let mut data = [0.; 4];
  gl::GetFloatv(gl::COLOR_CLEAR_VALUE, data.as_mut_ptr());
  Ok(data)
}

unsafe fn get_ctx_blending_state() -> Result<BlendingState, StateQueryError> {
  let state = gl::IsEnabled(gl::BLEND);

  match state {
    gl::TRUE => Ok(BlendingState::On),
    gl::FALSE => Ok(BlendingState::Off),
    _ => Err(StateQueryError::UnknownBlendingState(state)),
  }
}

unsafe fn get_ctx_blending_equation() -> Result<Equation, StateQueryError> {
  let mut data = gl::FUNC_ADD as GLint;
  gl::GetIntegerv(gl::BLEND_EQUATION_RGB, &mut data);

  let data = data as GLenum;
  match data {
    gl::FUNC_ADD => Ok(Equation::Additive),
    gl::FUNC_SUBTRACT => Ok(Equation::Subtract),
    gl::FUNC_REVERSE_SUBTRACT => Ok(Equation::ReverseSubtract),
    gl::MIN => Ok(Equation::Min),
    gl::MAX => Ok(Equation::Max),
    _ => Err(StateQueryError::UnknownBlendingEquation(data)),
  }
}

unsafe fn get_ctx_blending_factors() -> Result<(Factor, Factor), StateQueryError> {
  let mut src = gl::ONE as GLint;
  let mut dst = gl::ZERO as GLint;

  gl::GetIntegerv(gl::BLEND_SRC_RGB, &mut src);
  gl::GetIntegerv(gl::BLEND_DST_RGB, &mut dst);

  let src_k =
    from_gl_blending_factor(src as GLenum).map_err(StateQueryError::UnknownBlendingSrcFactor)?;
  let dst_k =
    from_gl_blending_factor(dst as GLenum).map_err(StateQueryError::UnknownBlendingDstFactor)?;

  Ok((src_k, dst_k))
}

#[inline]
fn from_gl_blending_factor(factor: GLenum) -> Result<Factor, GLenum> {
  match factor {
    gl::ONE => Ok(Factor::One),
    gl::ZERO => Ok(Factor::Zero),
    gl::SRC_COLOR => Ok(Factor::SrcColor),
    gl::ONE_MINUS_SRC_COLOR => Ok(Factor::SrcColorComplement),
    gl::DST_COLOR => Ok(Factor::DestColor),
    gl::ONE_MINUS_DST_COLOR => Ok(Factor::DestColorComplement),
    gl::SRC_ALPHA => Ok(Factor::SrcAlpha),
    gl::ONE_MINUS_SRC_ALPHA => Ok(Factor::SrcAlphaComplement),
    gl::DST_ALPHA => Ok(Factor::DstAlpha),
    gl::ONE_MINUS_DST_ALPHA => Ok(Factor::DstAlphaComplement),
    gl::SRC_ALPHA_SATURATE => Ok(Factor::SrcAlphaSaturate),
    _ => Err(factor),
  }
}

unsafe fn get_ctx_depth_test() -> Result<DepthTest, StateQueryError> {
  let state = gl::IsEnabled(gl::DEPTH_TEST);

  match state {
    gl::TRUE => Ok(DepthTest::On),
    gl::FALSE => Ok(DepthTest::Off),
    _ => Err(StateQueryError::UnknownDepthTestState(state)),
  }
}

unsafe fn get_ctx_face_culling_state() -> Result<FaceCullingState, StateQueryError> {
  let state = gl::IsEnabled(gl::CULL_FACE);

  match state {
    gl::TRUE => Ok(FaceCullingState::On),
    gl::FALSE => Ok(FaceCullingState::Off),
    _ => Err(StateQueryError::UnknownFaceCullingState(state)),
  }
}

unsafe fn get_ctx_face_culling_order() -> Result<FaceCullingOrder, StateQueryError> {
  let mut order = gl::CCW as GLint;
  gl::GetIntegerv(gl::FRONT_FACE, &mut order);

  let order = order as GLenum;
  match order {
    gl::CCW => Ok(FaceCullingOrder::CCW),
    gl::CW => Ok(FaceCullingOrder::CW),
    _ => Err(StateQueryError::UnknownFaceCullingOrder(order)),
  }
}

unsafe fn get_ctx_face_culling_mode() -> Result<FaceCullingMode, StateQueryError> {
  let mut mode = gl::BACK as GLint;
  gl::GetIntegerv(gl::CULL_FACE_MODE, &mut mode);

  let mode = mode as GLenum;
  match mode {
    gl::FRONT => Ok(FaceCullingMode::Front),
    gl::BACK => Ok(FaceCullingMode::Back),
    gl::FRONT_AND_BACK => Ok(FaceCullingMode::Both),
    _ => Err(StateQueryError::UnknownFaceCullingMode(mode)),
  }
}

unsafe fn get_ctx_vertex_restart() -> Result<VertexRestart, StateQueryError> {
  let state = gl::IsEnabled(gl::PRIMITIVE_RESTART);

  match state {
    gl::TRUE => Ok(VertexRestart::On),
    gl::FALSE => Ok(VertexRestart::Off),
    _ => Err(StateQueryError::UnknownVertexRestartState(state)),
  }
}

unsafe fn get_ctx_current_texture_unit() -> Result<GLenum, StateQueryError> {
  let mut active_texture = gl::TEXTURE0 as GLint;
  gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut active_texture);
  Ok(active_texture as GLenum)
}

unsafe fn get_ctx_bound_draw_framebuffer() -> Result<GLuint, StateQueryError> {
  let mut bound = 0 as GLint;
  gl::GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut bound);
  Ok(bound as GLuint)
}

unsafe fn get_ctx_bound_vertex_array() -> Result<GLuint, StateQueryError> {
  let mut bound = 0 as GLint;
  gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut bound);
  Ok(bound as GLuint)
}

unsafe fn get_ctx_current_program() -> Result<GLuint, StateQueryError> {
  let mut used = 0 as GLint;
  gl::GetIntegerv(gl::CURRENT_PROGRAM, &mut used);
  Ok(used as GLuint)
}

unsafe fn get_ctx_srgb_framebuffer_enabled() -> Result<bool, StateQueryError> {
  let state = gl::IsEnabled(gl::FRAMEBUFFER_SRGB);

  match state {
    gl::TRUE => Ok(true),
    gl::FALSE => Ok(false),
    _ => Err(StateQueryError::UnknownSRGBFramebufferState(state)),
  }
}
