use crate::backend::shading_gate::ShadingGate as ShadingGateBackend;
use crate::context::GraphicsContext;
use crate::render_gate::RenderGate;
use crate::shader::{Program, ProgramInterface, UniformInterface};
use crate::vertex::Semantics;

pub struct ShadingGate<'a, C>
where
  C: ?Sized + GraphicsContext,
  C::Backend: ShadingGateBackend,
{
  pub(crate) ctx: &'a mut C,
}

impl<'a, C> ShadingGate<'a, C>
where
  C: ?Sized + GraphicsContext,
  C::Backend: ShadingGateBackend,
{
  pub fn shade<Sem, Out, Uni, F>(&mut self, program: &mut Program<C::Backend, Sem, Out, Uni>, f: F)
  where
    Sem: Semantics,
    Uni: UniformInterface<C::Backend>,
    F: for<'b> FnOnce(ProgramInterface<'b, C::Backend, Uni>, RenderGate<'b, C>),
  {
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
