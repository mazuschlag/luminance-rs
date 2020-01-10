//! Shader API.

use std::marker::PhantomData;

use crate::backend::shader::{
  ProgramError, ProgramWarning, Shader, StageError, StageType, TessellationStages, Uniform,
  UniformWarning, Uniformable,
};
use crate::context::GraphicsContext;
use crate::vertex::Semantics;

pub struct Stage<S>
where
  S: Shader,
{
  repr: S::StageRepr,
}

impl<S> Stage<S>
where
  S: Shader,
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
  S: Shader,
{
  fn drop(&mut self) {
    unsafe { S::destroy_stage(&mut self.repr) }
  }
}

pub struct UniformBuilder<'a, S>
where
  S: Shader,
{
  repr: S::UniformBuilderRepr,
  warnings: Vec<UniformWarning>,
  _a: PhantomData<&'a mut ()>,
}

impl<'a, S> UniformBuilder<'a, S>
where
  S: Shader,
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

pub trait UniformInterface<E = ()>: Sized {
  fn uniform_interface<'a, S>(
    builder: &mut UniformBuilder<'a, S>,
    env: &mut E,
  ) -> Result<Self, UniformWarning>
  where
    S: Shader;
}

impl UniformInterface for () {
  fn uniform_interface<'a, S>(
    _: &mut UniformBuilder<'a, S>,
    _: &mut (),
  ) -> Result<Self, UniformWarning>
  where
    S: Shader,
  {
    Ok(())
  }
}

/// A built program with potential warnings.
///
/// The sole purpose of this type is to be destructured when a program is built.
pub struct BuiltProgram<S, Sem, Out, Uni>
where
  S: Shader,
{
  /// Built program.
  pub program: Program<S, Sem, Out, Uni>,
  /// Potential warnings.
  pub warnings: Vec<ProgramError>,
}

impl<S, Sem, Out, Uni> BuiltProgram<S, Sem, Out, Uni>
where
  S: Shader,
{
  /// Get the program and ignore the warnings.
  pub fn ignore_warnings(self) -> Program<S, Sem, Out, Uni> {
    self.program
  }
}

/// A [`Program`] uniform adaptation that has failed.
pub struct AdaptationFailure<S, Sem, Out, Uni>
where
  S: Shader,
{
  /// Program used before trying to adapt.
  pub program: Program<S, Sem, Out, Uni>,
  /// Program error that prevented to adapt.
  pub error: ProgramError,
}

impl<S, Sem, Out, Uni> AdaptationFailure<S, Sem, Out, Uni>
where
  S: Shader,
{
  /// Get the program and ignore the error.
  pub fn ignore_error(self) -> Program<S, Sem, Out, Uni> {
    self.program
  }
}

pub struct Program<S, Sem, Out, Uni>
where
  S: Shader,
{
  repr: S::ProgramRepr,
  uni: Uni,
  _sem: PhantomData<*const Sem>,
  _out: PhantomData<*const Out>,
}

impl<S, Sem, Out, Uni> Drop for Program<S, Sem, Out, Uni>
where
  S: Shader,
{
  fn drop(&mut self) {
    unsafe { S::destroy_program(&mut self.repr) }
  }
}

impl<S, Sem, Out, Uni> Program<S, Sem, Out, Uni>
where
  S: Shader,
  Sem: Semantics,
{
  pub fn from_stages_env<C, T, G, E>(
    ctx: &mut C,
    vertex: &Stage<S>,
    tess: T,
    geometry: G,
    fragment: &Stage<S>,
    env: &mut E,
  ) -> Result<BuiltProgram<S, Sem, Out, Uni>, ProgramError>
  where
    C: GraphicsContext<Backend = S>,
    Uni: UniformInterface<E>,
    T: for<'a> Into<Option<TessellationStages<'a, Stage<S>>>>,
    G: for<'a> Into<Option<&'a Stage<S>>>,
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
        .map(|w| ProgramError::Warning(ProgramWarning::VertexAttrib(w)))
        .collect();

      let mut uniform_builder: UniformBuilder<S> =
        S::new_uniform_builder(&mut repr).map(|repr| UniformBuilder {
          repr,
          warnings: Vec::new(),
          _a: PhantomData,
        })?;
      let uni = Uni::uniform_interface(&mut uniform_builder, env)
        .map_err(|w| ProgramError::Warning(ProgramWarning::Uniform(w)))?;

      let program = Program {
        repr,
        uni,
        _sem: PhantomData,
        _out: PhantomData,
      };

      Ok(BuiltProgram { program, warnings })
    }
  }
}
