use crate::render_state::RenderState;

pub trait RenderGate {
  fn enter_render_state(&mut self, rdr_st: &RenderState);
}
