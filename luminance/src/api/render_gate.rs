use crate::api::tess_gate::TessGate;
use crate::backend::render_gate::RenderGate as RenderGateBackend;
use crate::context::GraphicsContext;
use crate::render_state::RenderState;

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
  pub fn render<'b, R, F>(&'b mut self, rdr_st: R, f: F)
  where
    R: AsRef<RenderState>,
    F: FnOnce(TessGate<'b, C>),
  {
    unsafe {
      self.ctx.backend().enter_render_state(rdr_st.as_ref());
    }

    let tess_gate = TessGate { ctx: self.ctx };
    f(tess_gate);
  }
}
