//! Shader backend.

use crate::shader::{
  ProgramError, StageError, StageType, TessellationStages, Uniform, UniformType, UniformWarning,
  VertexAttribWarning,
};
use crate::vertex::Semantics;

pub unsafe trait Uniformable<S>
where
  S: ?Sized + Shader,
{
  unsafe fn ty() -> UniformType;

  unsafe fn update(&self, program: &mut S::ProgramRepr, uniform: &mut Uniform<Self>);
}

pub unsafe trait Shader {
  type StageRepr;

  type ProgramRepr;

  type UniformBuilderRepr;

  unsafe fn new_stage(&mut self, ty: StageType, src: &str) -> Result<Self::StageRepr, StageError>;

  unsafe fn destroy_stage(stage: &mut Self::StageRepr);

  unsafe fn new_program(
    &mut self,
    vertex: &Self::StageRepr,
    tess: Option<TessellationStages<Self::StageRepr>>,
    geometry: Option<&Self::StageRepr>,
    fragment: &Self::StageRepr,
  ) -> Result<Self::ProgramRepr, ProgramError>;

  unsafe fn destroy_program(program: &mut Self::ProgramRepr);

  unsafe fn apply_semantics<Sem>(
    program: &mut Self::ProgramRepr,
  ) -> Result<Vec<VertexAttribWarning>, ProgramError>
  where
    Sem: Semantics;

  unsafe fn new_uniform_builder(
    program: &mut Self::ProgramRepr,
  ) -> Result<Self::UniformBuilderRepr, ProgramError>;

  unsafe fn ask_uniform<T>(
    uniform_builder: &mut Self::UniformBuilderRepr,
    name: &str,
  ) -> Result<Uniform<T>, UniformWarning>
  where
    T: Uniformable<Self>;

  unsafe fn unbound<T>(uniform_builder: &mut Self::UniformBuilderRepr) -> Uniform<T>
  where
    T: Uniformable<Self>;
}
