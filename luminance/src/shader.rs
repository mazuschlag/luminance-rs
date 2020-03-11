//! Shader API.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::backend::shader::{Shader, Uniformable};
use crate::context::GraphicsContext;
use crate::vertex::Semantics;

/// A shader stage type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageType {
  /// Vertex shader.
  VertexShader,
  /// Tessellation control shader.
  TessellationControlShader,
  /// Tessellation evaluation shader.
  TessellationEvaluationShader,
  /// Geometry shader.
  GeometryShader,
  /// Fragment shader.
  FragmentShader,
}

impl fmt::Display for StageType {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageType::VertexShader => f.write_str("vertex shader"),
      StageType::TessellationControlShader => f.write_str("tessellation control shader"),
      StageType::TessellationEvaluationShader => f.write_str("tessellation evaluation shader"),
      StageType::GeometryShader => f.write_str("geometry shader"),
      StageType::FragmentShader => f.write_str("fragment shader"),
    }
  }
}

/// Errors that shader stages can emit.
#[derive(Clone, Debug)]
pub enum StageError {
  /// Occurs when a shader fails to compile.
  CompilationFailed(StageType, String),
  /// Occurs when you try to create a shader which type is not supported on the current hardware.
  UnsupportedType(StageType),
}

impl fmt::Display for StageError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageError::CompilationFailed(ref ty, ref r) => write!(f, "{} compilation error: {}", ty, r),

      StageError::UnsupportedType(ty) => write!(f, "unsupported {}", ty),
    }
  }
}

impl From<StageError> for ProgramError {
  fn from(e: StageError) -> Self {
    ProgramError::StageError(e)
  }
}

pub struct TessellationStages<'a, S>
where
  S: ?Sized,
{
  pub control: &'a S,
  pub evaluation: &'a S,
}

/// Errors that a `Program` can generate.
#[derive(Debug)]
pub enum ProgramError {
  /// A shader stage failed to compile or validate its state.
  StageError(StageError),
  /// Program link failed. You can inspect the reason by looking at the contained `String`.
  LinkFailed(String),
  Warning(ProgramWarning),
}

impl fmt::Display for ProgramError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      ProgramError::StageError(ref e) => write!(f, "shader program has stage error: {}", e),

      ProgramError::LinkFailed(ref s) => write!(f, "shader program failed to link: {}", s),

      ProgramError::Warning(ref e) => write!(f, "shader program warning: {}", e),
    }
  }
}

/// Program warnings, not necessarily considered blocking errors.
#[derive(Debug)]
pub enum ProgramWarning {
  /// Some uniform configuration is ill-formed. It can be a problem of inactive uniform, mismatch
  /// type, etc. Check the `UniformWarning` type for more information.
  Uniform(UniformWarning),
  /// Some vertex attribute is ill-formed.
  VertexAttrib(VertexAttribWarning),
}

impl fmt::Display for ProgramWarning {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      ProgramWarning::Uniform(ref e) => write!(f, "uniform warning: {}", e),
      ProgramWarning::VertexAttrib(ref e) => write!(f, "vertex attribute warning: {}", e),
    }
  }
}

impl From<ProgramWarning> for ProgramError {
  fn from(e: ProgramWarning) -> Self {
    ProgramError::Warning(e)
  }
}

/// Warnings related to uniform issues.
#[derive(Debug)]
pub enum UniformWarning {
  /// Inactive uniform (not in use / no participation to the final output in shaders).
  Inactive(String),
  /// Type mismatch between the static requested type (i.e. the `T` in [`Uniform<T>`] for instance)
  /// and the type that got reflected from the backend in the shaders.
  ///
  /// The first `String` is the name of the uniform; the second one gives the type mismatch.
  TypeMismatch(String, UniformType),
}

impl UniformWarning {
  /// Create an inactive uniform warning.
  pub fn inactive<N>(name: N) -> Self
  where
    N: Into<String>,
  {
    UniformWarning::Inactive(name.into())
  }

  /// Create a type mismatch.
  pub fn type_mismatch<N>(name: N, ty: UniformType) -> Self
  where
    N: Into<String>,
  {
    UniformWarning::TypeMismatch(name.into(), ty)
  }
}

impl fmt::Display for UniformWarning {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      UniformWarning::Inactive(ref s) => write!(f, "inactive {} uniform", s),

      UniformWarning::TypeMismatch(ref n, ref t) => {
        write!(f, "type mismatch for uniform {}: {}", n, t)
      }
    }
  }
}

impl From<UniformWarning> for ProgramWarning {
  fn from(e: UniformWarning) -> Self {
    ProgramWarning::Uniform(e)
  }
}

/// Warnings related to vertex attributes issues.
#[derive(Debug)]
pub enum VertexAttribWarning {
  /// Inactive vertex attribute (not read).
  Inactive(String),
}

impl fmt::Display for VertexAttribWarning {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      VertexAttribWarning::Inactive(ref s) => write!(f, "inactive {} vertex attribute", s),
    }
  }
}

impl From<VertexAttribWarning> for ProgramWarning {
  fn from(e: VertexAttribWarning) -> Self {
    ProgramWarning::VertexAttrib(e)
  }
}

#[derive(Debug)]
pub struct Uniform<T>
where
  T: ?Sized,
{
  index: i32,
  _t: PhantomData<*const T>,
}

impl<T> Uniform<T>
where
  T: ?Sized,
{
  pub unsafe fn new(index: i32) -> Self {
    Uniform {
      index,
      _t: PhantomData,
    }
  }

  pub fn index(&self) -> i32 {
    self.index
  }
}

/// Type of a uniform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UniformType {
  // scalars
  /// 32-bit signed integer.
  Int,
  /// 32-bit unsigned integer.
  UInt,
  /// 32-bit floating-point number.
  Float,
  /// Boolean.
  Bool,

  // vectors
  /// 2D signed integral vector.
  IVec2,
  /// 3D signed integral vector.
  IVec3,
  /// 4D signed integral vector.
  IVec4,
  /// 2D unsigned integral vector.
  UIVec2,
  /// 3D unsigned integral vector.
  UIVec3,
  /// 4D unsigned integral vector.
  UIVec4,
  /// 2D floating-point vector.
  Vec2,
  /// 3D floating-point vector.
  Vec3,
  /// 4D floating-point vector.
  Vec4,
  /// 2D boolean vector.
  BVec2,
  /// 3D boolean vector.
  BVec3,
  /// 4D boolean vector.
  BVec4,

  // matrices
  /// 2×2 floating-point matrix.
  M22,
  /// 3×3 floating-point matrix.
  M33,
  /// 4×4 floating-point matrix.
  M44,

  // textures
  /// Signed integral 1D texture sampler.
  ISampler1D,
  /// Signed integral 2D texture sampler.
  ISampler2D,
  /// Signed integral 3D texture sampler.
  ISampler3D,
  /// Signed integral 1D array texture sampler.
  ISampler1DArray,
  /// Signed integral 2D array texture sampler.
  ISampler2DArray,
  /// Unsigned integral 1D texture sampler.
  UISampler1D,
  /// Unsigned integral 2D texture sampler.
  UISampler2D,
  /// Unsigned integral 3D texture sampler.
  UISampler3D,
  /// Unsigned integral 1D array texture sampler.
  UISampler1DArray,
  /// Unsigned integral 2D array texture sampler.
  UISampler2DArray,
  /// Floating-point 1D texture sampler.
  Sampler1D,
  /// Floating-point 2D texture sampler.
  Sampler2D,
  /// Floating-point 3D texture sampler.
  Sampler3D,
  /// Floating-point 1D array texture sampler.
  Sampler1DArray,
  /// Floating-point 2D array texture sampler.
  Sampler2DArray,
  /// Signed cubemap sampler.
  ICubemap,
  /// Unsigned cubemap sampler.
  UICubemap,
  /// Floating-point cubemap sampler.
  Cubemap,

  // buffer
  /// Buffer binding; used for UBOs.
  BufferBinding,
}

impl fmt::Display for UniformType {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      UniformType::Int => f.write_str("int"),
      UniformType::UInt => f.write_str("uint"),
      UniformType::Float => f.write_str("float"),
      UniformType::Bool => f.write_str("bool"),
      UniformType::IVec2 => f.write_str("ivec2"),
      UniformType::IVec3 => f.write_str("ivec3"),
      UniformType::IVec4 => f.write_str("ivec4"),
      UniformType::UIVec2 => f.write_str("uvec2"),
      UniformType::UIVec3 => f.write_str("uvec3"),
      UniformType::UIVec4 => f.write_str("uvec4"),
      UniformType::Vec2 => f.write_str("vec2"),
      UniformType::Vec3 => f.write_str("vec3"),
      UniformType::Vec4 => f.write_str("vec4"),
      UniformType::BVec2 => f.write_str("bvec2"),
      UniformType::BVec3 => f.write_str("bvec3"),
      UniformType::BVec4 => f.write_str("bvec4"),
      UniformType::M22 => f.write_str("mat2"),
      UniformType::M33 => f.write_str("mat3"),
      UniformType::M44 => f.write_str("mat4"),
      UniformType::ISampler1D => f.write_str("isampler1D"),
      UniformType::ISampler2D => f.write_str("isampler2D"),
      UniformType::ISampler3D => f.write_str("isampler3D"),
      UniformType::ISampler1DArray => f.write_str("isampler1DArray"),
      UniformType::ISampler2DArray => f.write_str("isampler2DArray"),
      UniformType::UISampler1D => f.write_str("usampler1D"),
      UniformType::UISampler2D => f.write_str("usampler2D"),
      UniformType::UISampler3D => f.write_str("usampler3D"),
      UniformType::UISampler1DArray => f.write_str("usampler1DArray"),
      UniformType::UISampler2DArray => f.write_str("usampler2DArray"),
      UniformType::Sampler1D => f.write_str("sampler1D"),
      UniformType::Sampler2D => f.write_str("sampler2D"),
      UniformType::Sampler3D => f.write_str("sampler3D"),
      UniformType::Sampler1DArray => f.write_str("sampler1DArray"),
      UniformType::Sampler2DArray => f.write_str("sampler2DArray"),
      UniformType::ICubemap => f.write_str("isamplerCube"),
      UniformType::UICubemap => f.write_str("usamplerCube"),
      UniformType::Cubemap => f.write_str("samplerCube"),
      UniformType::BufferBinding => f.write_str("buffer binding"),
    }
  }
}

pub struct Stage<S>
where
  S: ?Sized + Shader,
{
  repr: S::StageRepr,
}

impl<S> Stage<S>
where
  S: ?Sized + Shader,
{
  pub fn new<C, R>(ctx: &mut C, ty: StageType, src: R) -> Result<Self, StageError>
  where
    C: GraphicsContext<Backend = S>,
    R: AsRef<str>,
  {
    unsafe {
      ctx
        .backend()
        .new_stage(ty, src.as_ref())
        .map(|repr| Stage { repr })
    }
  }
}

impl<S> Drop for Stage<S>
where
  S: ?Sized + Shader,
{
  fn drop(&mut self) {
    unsafe { S::destroy_stage(&mut self.repr) }
  }
}

pub struct UniformBuilder<'a, S>
where
  S: ?Sized + Shader,
{
  repr: S::UniformBuilderRepr,
  warnings: Vec<UniformWarning>,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, S> UniformBuilder<'a, S>
where
  S: ?Sized + Shader,
{
  pub fn ask<T, N>(&mut self, name: N) -> Result<Uniform<T>, UniformWarning>
  where
    N: AsRef<str>,
    T: Uniformable<S>,
  {
    unsafe { S::ask_uniform(&mut self.repr, name.as_ref()) }
  }

  pub fn ask_or_unbound<T, N>(&mut self, name: N) -> Uniform<T>
  where
    N: AsRef<str>,
    T: Uniformable<S>,
  {
    match self.ask(name) {
      Ok(uniform) => uniform,
      Err(err) => {
        self.warnings.push(err);
        unsafe { S::unbound(&mut self.repr) }
      }
    }
  }
}

pub trait UniformInterface<S, E = ()>: Sized
where
  S: ?Sized + Shader,
{
  fn uniform_interface<'a>(
    builder: &mut UniformBuilder<'a, S>,
    env: &mut E,
  ) -> Result<Self, UniformWarning>;
}

impl<S, E> UniformInterface<S, E> for ()
where
  S: ?Sized + Shader,
{
  fn uniform_interface<'a>(
    _: &mut UniformBuilder<'a, S>,
    _: &mut E,
  ) -> Result<Self, UniformWarning> {
    Ok(())
  }
}

/// A built program with potential warnings.
///
/// The sole purpose of this type is to be destructured when a program is built.
pub struct BuiltProgram<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
{
  /// Built program.
  pub program: Program<S, Sem, Out, Uni>,
  /// Potential warnings.
  pub warnings: Vec<ProgramError>,
}

impl<S, Sem, Out, Uni> BuiltProgram<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
{
  /// Get the program and ignore the warnings.
  pub fn ignore_warnings(self) -> Program<S, Sem, Out, Uni> {
    self.program
  }
}

/// A [`Program`] uniform adaptation that has failed.
pub struct AdaptationFailure<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
{
  /// Program used before trying to adapt.
  pub program: Program<S, Sem, Out, Uni>,
  /// Program error that prevented to adapt.
  pub error: ProgramError,
}

impl<S, Sem, Out, Uni> AdaptationFailure<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
{
  pub(crate) fn new(program: Program<S, Sem, Out, Uni>, error: ProgramError) -> Self {
    AdaptationFailure { program, error }
  }

  /// Get the program and ignore the error.
  pub fn ignore_error(self) -> Program<S, Sem, Out, Uni> {
    self.program
  }
}

pub struct ProgramInterface<'a, S, Uni>
where
  S: ?Sized + Shader,
{
  pub(crate) program: &'a mut S::ProgramRepr,
  pub(crate) uni: &'a Uni,
}

impl<'a, S, Uni> Deref for ProgramInterface<'a, S, Uni>
where
  S: ?Sized + Shader,
{
  type Target = Uni;

  fn deref(&self) -> &Self::Target {
    self.uni
  }
}

impl<'a, S, Uni> ProgramInterface<'a, S, Uni>
where
  S: ?Sized + Shader,
{
  pub fn set<T>(&mut self, uniform: &Uniform<T>, value: T)
  where
    T: Uniformable<S>,
  {
    unsafe { T::update(value, self.program, uniform) };
  }

  pub fn query(&'a mut self) -> Result<UniformBuilder<'a, S>, ProgramError> {
    unsafe {
      S::new_uniform_builder(&mut self.program).map(|repr| UniformBuilder {
        repr,
        warnings: Vec::new(),
        _a: PhantomData,
      })
    }
  }
}

pub struct Program<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
{
  pub(crate) repr: S::ProgramRepr,
  pub(crate) uni: Uni,
  _sem: PhantomData<*const Sem>,
  _out: PhantomData<*const Out>,
}

impl<S, Sem, Out, Uni> Drop for Program<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
{
  fn drop(&mut self) {
    unsafe { S::destroy_program(&mut self.repr) }
  }
}

impl<S, Sem, Out, Uni> Program<S, Sem, Out, Uni>
where
  S: ?Sized + Shader,
  Sem: Semantics,
{
  pub fn from_stages_env<'a, C, T, G, E>(
    ctx: &mut C,
    vertex: &'a Stage<S>,
    tess: T,
    geometry: G,
    fragment: &'a Stage<S>,
    env: &mut E,
  ) -> Result<BuiltProgram<S, Sem, Out, Uni>, ProgramError>
  where
    C: GraphicsContext<Backend = S>,
    Uni: UniformInterface<S, E>,
    T: Into<Option<TessellationStages<'a, Stage<S>>>>,
    G: Into<Option<&'a Stage<S>>>,
  {
    let tess = tess.into();
    let geometry = geometry.into();

    unsafe {
      let mut repr = ctx.backend().new_program(
        &vertex.repr,
        tess.map(|stages| TessellationStages {
          control: &stages.control.repr,
          evaluation: &stages.evaluation.repr,
        }),
        geometry.map(|stage| &stage.repr),
        &fragment.repr,
      )?;

      let warnings = S::apply_semantics::<Sem>(&mut repr)?
        .into_iter()
        .map(|w| ProgramError::Warning(w.into()))
        .collect();

      let mut uniform_builder: UniformBuilder<S> =
        S::new_uniform_builder(&mut repr).map(|repr| UniformBuilder {
          repr,
          warnings: Vec::new(),
          _a: PhantomData,
        })?;

      let uni =
        Uni::uniform_interface(&mut uniform_builder, env).map_err(ProgramWarning::Uniform)?;

      let program = Program {
        repr,
        uni,
        _sem: PhantomData,
        _out: PhantomData,
      };

      Ok(BuiltProgram { program, warnings })
    }
  }

  pub fn from_stages<C, T, G>(
    ctx: &mut C,
    vertex: &Stage<S>,
    tess: T,
    geometry: G,
    fragment: &Stage<S>,
  ) -> Result<BuiltProgram<S, Sem, Out, Uni>, ProgramError>
  where
    C: GraphicsContext<Backend = S>,
    Uni: UniformInterface<S>,
    T: for<'a> Into<Option<TessellationStages<'a, Stage<S>>>>,
    G: for<'a> Into<Option<&'a Stage<S>>>,
  {
    Self::from_stages_env(ctx, vertex, tess, geometry, fragment, &mut ())
  }

  pub fn from_strings_env<'a, C, V, T, G, F, E>(
    ctx: &mut C,
    vertex: V,
    tess: T,
    geometry: G,
    fragment: F,
    env: &mut E,
  ) -> Result<BuiltProgram<S, Sem, Out, Uni>, ProgramError>
  where
    C: GraphicsContext<Backend = S>,
    Uni: UniformInterface<S, E>,
    V: AsRef<str> + 'a,
    T: Into<Option<TessellationStages<'a, str>>>,
    G: Into<Option<&'a str>>,
    F: AsRef<str> + 'a,
  {
    let vs_stage = Stage::new(ctx, StageType::VertexShader, vertex)?;

    let tess_stages = match tess.into() {
      Some(TessellationStages {
        control,
        evaluation,
      }) => {
        let control_stage = Stage::new(ctx, StageType::TessellationControlShader, control)?;
        let evaluation_stage =
          Stage::new(ctx, StageType::TessellationEvaluationShader, evaluation)?;
        Some((control_stage, evaluation_stage))
      }
      None => None,
    };
    let tess_stages =
      tess_stages
        .as_ref()
        .map(|(ref control, ref evaluation)| TessellationStages {
          control,
          evaluation,
        });

    let gs_stage = match geometry.into() {
      Some(geometry) => Some(Stage::new(ctx, StageType::GeometryShader, geometry)?),
      None => None,
    };

    let fs_stage = Stage::new(ctx, StageType::FragmentShader, fragment)?;

    Self::from_stages_env(
      ctx,
      &vs_stage,
      tess_stages,
      gs_stage.as_ref(),
      &fs_stage,
      env,
    )
  }

  pub fn from_strings<'a, C, V, T, G, F>(
    ctx: &mut C,
    vertex: V,
    tess: T,
    geometry: G,
    fragment: F,
  ) -> Result<BuiltProgram<S, Sem, Out, Uni>, ProgramError>
  where
    C: GraphicsContext<Backend = S>,
    Uni: UniformInterface<S>,
    V: AsRef<str> + 'a,
    T: Into<Option<TessellationStages<'a, str>>>,
    G: Into<Option<&'a str>>,
    F: AsRef<str> + 'a,
  {
    Self::from_strings_env(ctx, vertex, tess, geometry, fragment, &mut ())
  }

  pub fn adapt<Q>(self) -> Result<BuiltProgram<S, Sem, Out, Q>, AdaptationFailure<S, Sem, Out, Uni>>
  where
    Q: UniformInterface<S>,
  {
    self.adapt_env(&mut ())
  }

  pub fn adapt_env<Q, E>(
    mut self,
    env: &mut E,
  ) -> Result<BuiltProgram<S, Sem, Out, Q>, AdaptationFailure<S, Sem, Out, Uni>>
  where
    Q: UniformInterface<S, E>,
  {
    // first, try to create the new uniform interface
    let mut uniform_builder: UniformBuilder<S> =
      match unsafe { S::new_uniform_builder(&mut self.repr) } {
        Ok(repr) => UniformBuilder {
          repr,
          warnings: Vec::new(),
          _a: PhantomData,
        },

        Err(e) => return Err(AdaptationFailure::new(self, e)),
      };

    let uni = match Q::uniform_interface(&mut uniform_builder, env) {
      Ok(uni) => uni,
      Err(e) => {
        return Err(AdaptationFailure::new(
          self,
          ProgramWarning::Uniform(e).into(),
        ))
      }
    };

    let warnings = uniform_builder
      .warnings
      .into_iter()
      .map(|w| ProgramError::Warning(w.into()))
      .collect();

    // we need to forget self so that we can move-out repr
    let self_ = std::mem::ManuallyDrop::new(self);
    let repr = unsafe { std::ptr::read(&self_.repr) };

    let program = Program {
      repr,
      uni,
      _sem: PhantomData,
      _out: PhantomData,
    };

    Ok(BuiltProgram { program, warnings })
  }

  pub fn readapt_env<E>(
    self,
    env: &mut E,
  ) -> Result<BuiltProgram<S, Sem, Out, Uni>, AdaptationFailure<S, Sem, Out, Uni>>
  where
    Uni: UniformInterface<S, E>,
  {
    self.adapt_env(env)
  }
}
