use crate::backend::render_gate::RenderGate as RenderGateBackend;
use crate::context::GraphicsContext;
use crate::render_state::RenderState;
use crate::tess_gate::TessGate;

pub struct RenderGate<'a, C>
where
  C: ?Sized,
{
  pub(crate) ctx: &'a mut C,
}

impl<'a, C> RenderGate<'a, C>
where
  C: ?Sized + GraphicsContext,
  C::Backend: RenderGateBackend,
{
  pub fn render<'b, F>(&'b mut self, rdr_st: &RenderState, f: F)
  where
    F: FnOnce(TessGate<'b, C>),
  {
    unsafe {
      self.ctx.backend().enter_render_state(rdr_st);
    }

    let tess_gate = TessGate { ctx: self.ctx };
    f(tess_gate);
  }
}
