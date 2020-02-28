use gl;
use gl::types::*;
use std::ffi::CString;
use std::ptr::{null, null_mut};

use crate::gl33::pipeline::{BoundBuffer, BoundTexture};
use crate::gl33::GL33;
use luminance::backend::shader::{Shader, Uniformable};
use luminance::linear::{M22, M33, M44};
use luminance::pixel::{Pixel, SamplerType as _, Type as PixelType};
use luminance::shader::{
  ProgramError, StageError, StageType, TessellationStages, Uniform, UniformType, UniformWarning,
  VertexAttribWarning,
};
use luminance::texture::{Dim, Dimensionable, Layerable};
use luminance::vertex::Semantics;

#[derive(Debug)]
pub struct Stage {
  handle: GLuint,
  ty: StageType,
}

#[derive(Debug)]
pub struct Program {
  pub(crate) handle: GLuint,
}

impl Program {
  fn dup(&self) -> Self {
    Program {
      handle: self.handle,
    }
  }

  fn link(&self) -> Result<(), ProgramError> {
    let handle = self.handle;

    unsafe {
      gl::LinkProgram(handle);

      let mut linked: GLint = gl::FALSE.into();
      gl::GetProgramiv(handle, gl::LINK_STATUS, &mut linked);

      if linked == gl::TRUE.into() {
        Ok(())
      } else {
        let mut log_len: GLint = 0;
        gl::GetProgramiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

        let mut log: Vec<u8> = Vec::with_capacity(log_len as usize);
        gl::GetProgramInfoLog(handle, log_len, null_mut(), log.as_mut_ptr() as *mut GLchar);

        gl::DeleteProgram(handle);

        log.set_len(log_len as usize);

        Err(ProgramError::LinkFailed(String::from_utf8(log).unwrap()))
      }
    }
  }
}

pub struct UniformBuilder {
  program: Program,
}

impl UniformBuilder {
  fn new(program: Program) -> Self {
    UniformBuilder { program }
  }

  fn ask_uniform<T>(&self, name: &str) -> Result<Uniform<T>, UniformWarning>
  where
    T: Uniformable<GL33>,
  {
    let location = {
      let c_name = CString::new(name.as_bytes()).unwrap();
      unsafe { gl::GetUniformLocation(self.program.handle, c_name.as_ptr() as *const GLchar) }
    };

    if location < 0 {
      Err(UniformWarning::Inactive(name.to_owned()))
    } else {
      Ok(unsafe { Uniform::new(location) })
    }
  }

  fn ask_uniform_block<T>(&self, name: &str) -> Result<Uniform<T>, UniformWarning>
  where
    T: Uniformable<GL33>,
  {
    let location = {
      let c_name = CString::new(name.as_bytes()).unwrap();
      unsafe { gl::GetUniformBlockIndex(self.program.handle, c_name.as_ptr() as *const GLchar) }
    };

    if location == gl::INVALID_INDEX {
      Err(UniformWarning::Inactive(name.to_owned()))
    } else {
      Ok(unsafe { Uniform::new(location as _) })
    }
  }
}

unsafe impl Shader for GL33 {
  type StageRepr = Stage;

  type ProgramRepr = Program;

  type UniformBuilderRepr = UniformBuilder;

  unsafe fn new_stage(&mut self, ty: StageType, src: &str) -> Result<Self::StageRepr, StageError> {
    let handle = gl::CreateShader(opengl_shader_type(ty));

    if handle == 0 {
      return Err(StageError::CompilationFailed(
        ty,
        "unable to create shader stage".to_owned(),
      ));
    }

    let c_src = CString::new(glsl_pragma_src(src).as_bytes()).unwrap();
    gl::ShaderSource(handle, 1, [c_src.as_ptr()].as_ptr(), null());
    gl::CompileShader(handle);

    let mut compiled: GLint = gl::FALSE.into();
    gl::GetShaderiv(handle, gl::COMPILE_STATUS, &mut compiled);

    if compiled == gl::TRUE.into() {
      Ok(Stage { handle, ty })
    } else {
      let mut log_len: GLint = 0;
      gl::GetShaderiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

      let mut log: Vec<u8> = Vec::with_capacity(log_len as usize);
      gl::GetShaderInfoLog(handle, log_len, null_mut(), log.as_mut_ptr() as *mut GLchar);

      gl::DeleteShader(handle);

      log.set_len(log_len as usize);

      Err(StageError::CompilationFailed(
        ty,
        String::from_utf8(log).unwrap(),
      ))
    }
  }

  unsafe fn destroy_stage(stage: &mut Self::StageRepr) {
    gl::DeleteShader(stage.handle);
  }

  unsafe fn new_program(
    &mut self,
    vertex: &Self::StageRepr,
    tess: Option<TessellationStages<Self::StageRepr>>,
    geometry: Option<&Self::StageRepr>,
    fragment: &Self::StageRepr,
  ) -> Result<Self::ProgramRepr, ProgramError> {
    let handle = gl::CreateProgram();

    if let Some(TessellationStages {
      control,
      evaluation,
    }) = tess
    {
      gl::AttachShader(handle, control.handle);
      gl::AttachShader(handle, evaluation.handle);
    }

    gl::AttachShader(handle, vertex.handle);

    if let Some(geometry) = geometry {
      gl::AttachShader(handle, geometry.handle);
    }

    gl::AttachShader(handle, fragment.handle);

    let program = Program { handle };
    program.link().map(move |_| program)
  }

  unsafe fn destroy_program(program: &mut Self::ProgramRepr) {
    gl::DeleteProgram(program.handle)
  }

  unsafe fn apply_semantics<Sem>(
    program: &mut Self::ProgramRepr,
  ) -> Result<Vec<VertexAttribWarning>, ProgramError>
  where
    Sem: Semantics,
  {
    let warnings = bind_vertex_attribs_locations::<Sem>(program);

    program.link()?;

    Ok(warnings)
  }

  unsafe fn new_uniform_builder(
    program: &mut Self::ProgramRepr,
  ) -> Result<Self::UniformBuilderRepr, ProgramError> {
    Ok(UniformBuilder::new(program.dup()))
  }

  unsafe fn ask_uniform<T>(
    uniform_builder: &mut Self::UniformBuilderRepr,
    name: &str,
  ) -> Result<Uniform<T>, UniformWarning>
  where
    T: Uniformable<Self>,
  {
    let uniform = match T::ty() {
      UniformType::BufferBinding => uniform_builder.ask_uniform_block(name)?,
      _ => uniform_builder.ask_uniform(name)?,
    };

    uniform_type_match(uniform_builder.program.handle, name, T::ty())?;

    Ok(uniform)
  }

  unsafe fn unbound<T>(_: &mut Self::UniformBuilderRepr) -> Uniform<T>
  where
    T: Uniformable<Self>,
  {
    Uniform::new(-1)
  }
}

fn opengl_shader_type(t: StageType) -> GLenum {
  match t {
    StageType::TessellationControlShader => gl::TESS_CONTROL_SHADER,
    StageType::TessellationEvaluationShader => gl::TESS_EVALUATION_SHADER,
    StageType::VertexShader => gl::VERTEX_SHADER,
    StageType::GeometryShader => gl::GEOMETRY_SHADER,
    StageType::FragmentShader => gl::FRAGMENT_SHADER,
  }
}

const GLSL_PRAGMA: &str = "#version 330 core\n\
                           #extension GL_ARB_separate_shader_objects : require\n";

fn glsl_pragma_src(src: &str) -> String {
  let mut pragma = String::from(GLSL_PRAGMA);
  pragma.push_str(src);
  pragma
}

fn uniform_type_match(program: GLuint, name: &str, ty: UniformType) -> Result<(), UniformWarning> {
  let mut size: GLint = 0;
  let mut glty: GLuint = 0;

  unsafe {
    // get the max length of the returned names
    let mut max_len = 0;
    gl::GetProgramiv(program, gl::ACTIVE_UNIFORM_MAX_LENGTH, &mut max_len);

    // get the index of the uniform
    let mut index = 0;

    let c_name = CString::new(name.as_bytes()).unwrap();
    gl::GetUniformIndices(
      program,
      1,
      [c_name.as_ptr() as *const GLchar].as_ptr(),
      &mut index,
    );

    // get its size and type
    let mut name_ = Vec::<GLchar>::with_capacity(max_len as usize);
    gl::GetActiveUniform(
      program,
      index,
      max_len,
      null_mut(),
      &mut size,
      &mut glty,
      name_.as_mut_ptr(),
    );
  }

  // early-return if array – we don’t support them yet
  if size != 1 {
    return Ok(());
  }

  check_types_match(name, ty, glty)
}

#[allow(clippy::cognitive_complexity)]
fn check_types_match(name: &str, ty: UniformType, glty: GLuint) -> Result<(), UniformWarning> {
  match ty {
    // scalars
    UniformType::Int if glty != gl::INT => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::UInt if glty != gl::UNSIGNED_INT => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::Float if glty != gl::FLOAT => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::Bool if glty != gl::BOOL => Err(UniformWarning::type_mismatch(name, ty)),
    // vectors
    UniformType::IVec2 if glty != gl::INT_VEC2 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::IVec3 if glty != gl::INT_VEC3 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::IVec4 if glty != gl::INT_VEC4 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::UIVec2 if glty != gl::UNSIGNED_INT_VEC2 => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::UIVec3 if glty != gl::UNSIGNED_INT_VEC3 => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::UIVec4 if glty != gl::UNSIGNED_INT_VEC4 => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::Vec2 if glty != gl::FLOAT_VEC2 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::Vec3 if glty != gl::FLOAT_VEC3 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::Vec4 if glty != gl::FLOAT_VEC4 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::BVec2 if glty != gl::BOOL_VEC2 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::BVec3 if glty != gl::BOOL_VEC3 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::BVec4 if glty != gl::BOOL_VEC4 => Err(UniformWarning::type_mismatch(name, ty)),
    // matrices
    UniformType::M22 if glty != gl::FLOAT_MAT2 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::M33 if glty != gl::FLOAT_MAT3 => Err(UniformWarning::type_mismatch(name, ty)),
    UniformType::M44 if glty != gl::FLOAT_MAT4 => Err(UniformWarning::type_mismatch(name, ty)),
    // textures
    UniformType::ISampler1D if glty != gl::INT_SAMPLER_1D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::ISampler2D if glty != gl::INT_SAMPLER_2D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::ISampler3D if glty != gl::INT_SAMPLER_3D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::UISampler1D if glty != gl::UNSIGNED_INT_SAMPLER_1D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::UISampler2D if glty != gl::UNSIGNED_INT_SAMPLER_2D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::UISampler3D if glty != gl::UNSIGNED_INT_SAMPLER_3D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::Sampler1D if glty != gl::SAMPLER_1D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::Sampler2D if glty != gl::SAMPLER_2D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::Sampler3D if glty != gl::SAMPLER_3D => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::ICubemap if glty != gl::INT_SAMPLER_CUBE => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::UICubemap if glty != gl::UNSIGNED_INT_SAMPLER_CUBE => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    UniformType::Cubemap if glty != gl::SAMPLER_CUBE => {
      Err(UniformWarning::type_mismatch(name, ty))
    }
    _ => Ok(()),
  }
}

fn bind_vertex_attribs_locations<Sem>(program: &Program) -> Vec<VertexAttribWarning>
where
  Sem: Semantics,
{
  let mut warnings = Vec::new();

  for desc in Sem::semantics_set() {
    match get_vertex_attrib_location(program, &desc.name) {
      Ok(_) => {
        let index = desc.index as GLuint;

        // we are not interested in the location as we’re about to change it to what we’ve
        // decided in the semantics
        let c_name = CString::new(desc.name.as_bytes()).unwrap();
        unsafe { gl::BindAttribLocation(program.handle, index, c_name.as_ptr() as *const GLchar) };
      }

      Err(warning) => warnings.push(warning),
    }
  }

  warnings
}

fn get_vertex_attrib_location(
  program: &Program,
  name: &str,
) -> Result<GLuint, VertexAttribWarning> {
  let location = {
    let c_name = CString::new(name.as_bytes()).unwrap();
    unsafe { gl::GetAttribLocation(program.handle, c_name.as_ptr() as *const GLchar) }
  };

  if location < 0 {
    Err(VertexAttribWarning::Inactive(name.to_owned()))
  } else {
    Ok(location as _)
  }
}

macro_rules! impl_Uniformable {
  (&[[$t:ty; $dim:expr]], $uty:tt, $f:tt) => {
    unsafe impl<'a> Uniformable<GL33> for &'a [[$t; $dim]] {
      unsafe fn ty() -> UniformType {
        UniformType::$uty
      }

      unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
        gl::$f(uniform.index(), self.len() as GLsizei, self.as_ptr() as _);
      }
    }
  };

  (&[$t:ty], $uty:tt, $f:tt) => {
    unsafe impl<'a> Uniformable<GL33> for &'a [$t] {
      unsafe fn ty() -> UniformType {
        UniformType::$uty
      }

      unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
        gl::$f(uniform.index(), self.len() as GLsizei, self.as_ptr());
      }
    }
  };

  ([$t:ty; $dim:expr], $uty:tt, $f:tt) => {
    unsafe impl Uniformable<GL33> for [$t; $dim] {
      unsafe fn ty() -> UniformType {
        UniformType::$uty
      }

      unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
        gl::$f(uniform.index(), 1, &self as _);
      }
    }
  };

  ($t:ty, $uty:tt, $f:tt) => {
    unsafe impl Uniformable<GL33> for $t {
      unsafe fn ty() -> UniformType {
        UniformType::$uty
      }

      unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
        gl::$f(uniform.index(), self);
      }
    }
  };

  // matrix notation
  (mat $t:ty, $uty:tt, $f:tt) => {
    unsafe impl Uniformable<GL33> for $t {
      unsafe fn ty() -> UniformType {
        UniformType::$uty
      }

      unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
        gl::$f(uniform.index(), 1, gl::FALSE, self.as_ptr() as _);
      }
    }
  };

  (mat &[$t:ty], $uty:tt, $f:tt) => {
    unsafe impl<'a> Uniformable<GL33> for &'a [$t] {
      unsafe fn ty() -> UniformType {
        UniformType::$uty
      }

      unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
        gl::$f(
          uniform.index(),
          self.len() as GLsizei,
          gl::FALSE,
          self.as_ptr() as _,
        );
      }
    }
  };
}

impl_Uniformable!(i32, Int, Uniform1i);
impl_Uniformable!([i32; 2], IVec2, Uniform2iv);
impl_Uniformable!([i32; 3], IVec3, Uniform3iv);
impl_Uniformable!([i32; 4], IVec4, Uniform4iv);
impl_Uniformable!(&[i32], Int, Uniform1iv);
impl_Uniformable!(&[[i32; 2]], IVec2, Uniform2iv);
impl_Uniformable!(&[[i32; 3]], IVec3, Uniform3iv);
impl_Uniformable!(&[[i32; 4]], IVec4, Uniform4iv);

impl_Uniformable!(u32, UInt, Uniform1ui);
impl_Uniformable!([u32; 2], UIVec2, Uniform2uiv);
impl_Uniformable!([u32; 3], UIVec3, Uniform3uiv);
impl_Uniformable!([u32; 4], UIVec4, Uniform4uiv);
impl_Uniformable!(&[u32], UInt, Uniform1uiv);
impl_Uniformable!(&[[u32; 2]], UIVec2, Uniform2uiv);
impl_Uniformable!(&[[u32; 3]], UIVec3, Uniform3uiv);
impl_Uniformable!(&[[u32; 4]], UIVec4, Uniform4uiv);

impl_Uniformable!(f32, Float, Uniform1f);
impl_Uniformable!([f32; 2], Vec2, Uniform2fv);
impl_Uniformable!([f32; 3], Vec3, Uniform3fv);
impl_Uniformable!([f32; 4], Vec4, Uniform4fv);
impl_Uniformable!(&[f32], Float, Uniform1fv);
impl_Uniformable!(&[[f32; 2]], Vec2, Uniform2fv);
impl_Uniformable!(&[[f32; 3]], Vec3, Uniform3fv);
impl_Uniformable!(&[[f32; 4]], Vec4, Uniform4fv);

impl_Uniformable!(mat M22, M22, UniformMatrix2fv);
impl_Uniformable!(mat & [M22], M22, UniformMatrix2fv);

impl_Uniformable!(mat M33, M33, UniformMatrix3fv);
impl_Uniformable!(mat & [M33], M33, UniformMatrix3fv);

impl_Uniformable!(mat M44, M44, UniformMatrix4fv);
impl_Uniformable!(mat & [M44], M44, UniformMatrix4fv);

unsafe impl Uniformable<GL33> for bool {
  unsafe fn ty() -> UniformType {
    UniformType::Bool
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    gl::Uniform1ui(uniform.index(), self as u32);
  }
}

unsafe impl Uniformable<GL33> for [bool; 2] {
  unsafe fn ty() -> UniformType {
    UniformType::BVec2
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v = [self[0] as u32, self[1] as u32];
    gl::Uniform2uiv(uniform.index(), 1, v.as_ptr() as _);
  }
}

unsafe impl Uniformable<GL33> for [bool; 3] {
  unsafe fn ty() -> UniformType {
    UniformType::BVec3
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v = [self[0] as u32, self[1] as u32, self[2] as u32];
    gl::Uniform3uiv(uniform.index(), 1, v.as_ptr() as _);
  }
}

unsafe impl Uniformable<GL33> for [bool; 4] {
  unsafe fn ty() -> UniformType {
    UniformType::BVec4
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v = [
      self[0] as u32,
      self[1] as u32,
      self[2] as u32,
      self[3] as u32,
    ];
    gl::Uniform4uiv(uniform.index(), 1, v.as_ptr() as _);
  }
}

unsafe impl<'a> Uniformable<GL33> for &'a [bool] {
  unsafe fn ty() -> UniformType {
    UniformType::Bool
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v: Vec<_> = self.iter().map(|x| *x as u32).collect();

    gl::Uniform1uiv(uniform.index(), v.len() as GLsizei, v.as_ptr() as _);
  }
}

unsafe impl<'a> Uniformable<GL33> for &'a [[bool; 2]] {
  unsafe fn ty() -> UniformType {
    UniformType::BVec2
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v: Vec<_> = self.iter().map(|x| [x[0] as u32, x[1] as u32]).collect();

    gl::Uniform2uiv(uniform.index(), v.len() as GLsizei, v.as_ptr() as _);
  }
}

unsafe impl<'a> Uniformable<GL33> for &'a [[bool; 3]] {
  unsafe fn ty() -> UniformType {
    UniformType::BVec3
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v: Vec<_> = self
      .iter()
      .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32])
      .collect();

    gl::Uniform3uiv(uniform.index(), v.len() as GLsizei, v.as_ptr() as _);
  }
}

unsafe impl<'a> Uniformable<GL33> for &'a [[bool; 4]] {
  unsafe fn ty() -> UniformType {
    UniformType::BVec4
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    let v: Vec<_> = self
      .iter()
      .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32, x[3] as u32])
      .collect();

    gl::Uniform4uiv(uniform.index(), v.len() as GLsizei, v.as_ptr() as _);
  }
}

unsafe impl Uniformable<GL33> for BoundBuffer {
  unsafe fn ty() -> UniformType {
    UniformType::BufferBinding
  }

  unsafe fn update(self, program: &mut Program, uniform: &mut Uniform<Self>) {
    gl::UniformBlockBinding(
      program.handle,
      uniform.index() as GLuint,
      self.binding as GLuint,
    )
  }
}

unsafe impl<L, D, P> Uniformable<GL33> for BoundTexture<L, D, P>
where
  L: Layerable,
  D: Dimensionable,
  P: Pixel,
{
  unsafe fn ty() -> UniformType {
    match (P::SamplerType::sample_type(), D::dim()) {
      (PixelType::NormIntegral, Dim::Dim1) => UniformType::Sampler1D,
      (PixelType::NormUnsigned, Dim::Dim1) => UniformType::Sampler1D,
      (PixelType::Integral, Dim::Dim1) => UniformType::ISampler1D,
      (PixelType::Unsigned, Dim::Dim1) => UniformType::UISampler1D,
      (PixelType::Floating, Dim::Dim1) => UniformType::Sampler1D,

      (PixelType::NormIntegral, Dim::Dim2) => UniformType::Sampler2D,
      (PixelType::NormUnsigned, Dim::Dim2) => UniformType::Sampler2D,
      (PixelType::Integral, Dim::Dim2) => UniformType::ISampler2D,
      (PixelType::Unsigned, Dim::Dim2) => UniformType::UISampler2D,
      (PixelType::Floating, Dim::Dim2) => UniformType::Sampler2D,

      (PixelType::NormIntegral, Dim::Dim3) => UniformType::Sampler3D,
      (PixelType::NormUnsigned, Dim::Dim3) => UniformType::Sampler3D,
      (PixelType::Integral, Dim::Dim3) => UniformType::ISampler3D,
      (PixelType::Unsigned, Dim::Dim3) => UniformType::UISampler3D,
      (PixelType::Floating, Dim::Dim3) => UniformType::Sampler3D,

      (PixelType::NormIntegral, Dim::Cubemap) => UniformType::Cubemap,
      (PixelType::NormUnsigned, Dim::Cubemap) => UniformType::Cubemap,
      (PixelType::Integral, Dim::Cubemap) => UniformType::ICubemap,
      (PixelType::Unsigned, Dim::Cubemap) => UniformType::UICubemap,
      (PixelType::Floating, Dim::Cubemap) => UniformType::Cubemap,
    }
  }

  unsafe fn update(self, _: &mut Program, uniform: &mut Uniform<Self>) {
    gl::Uniform1i(uniform.index(), self.unit as GLint)
  }
}
