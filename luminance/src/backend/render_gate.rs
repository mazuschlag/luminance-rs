use crate::render_state::RenderState;

pub unsafe trait RenderGate {
  unsafe fn enter_render_state(&mut self, rdr_st: &RenderState);
}
