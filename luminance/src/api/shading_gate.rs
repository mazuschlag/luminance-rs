use crate::api::render_gate::RenderGate;
use crate::api::shader::{Program, ProgramInterface, UniformInterface};
use crate::backend::shading_gate::ShadingGate as ShadingGateBackend;
use crate::context::GraphicsContext;
use crate::vertex::Semantics;

pub struct ShadingGate<'a, C>
where
  C: ?Sized + GraphicsContext,
  C::Backend: ShadingGateBackend,
{
  ctx: &'a mut C,
}

impl<'a, C> ShadingGate<'a, C>
where
  C: ?Sized + GraphicsContext,
  C::Backend: ShadingGateBackend,
{
  pub fn shade<Sem, Out, Uni, P, F>(&mut self, mut program: P, f: F)
  where
    P: AsMut<Program<C::Backend, Sem, Out, Uni>>,
    Sem: Semantics,
    Uni: UniformInterface,
    F: for<'b> FnOnce(ProgramInterface<'b, C::Backend, Uni>, RenderGate<'b, C>),
  {
    let program = program.as_mut();

    unsafe {
      self.ctx.backend().apply_shader_program(&mut program.repr);
    }

    let render_gate = RenderGate { ctx: self.ctx };
    let program_interface = ProgramInterface {
      program: &mut program.repr,
      uni: &program.uni,
    };

    f(program_interface, render_gate);
  }
}
