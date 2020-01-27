use crate::api::tess::TessView;
use crate::backend::tess_gate::TessGate as TessGateBackend;
use crate::context::GraphicsContext;

pub struct TessGate<'a, C>
where
  C: ?Sized,
{
  ctx: &'a mut C,
}

impl<'a, C> TessGate<'a, C>
where
  C: ?Sized + GraphicsContext,
  C::Backend: TessGateBackend,
{
  pub fn render<'b, T>(&'b mut self, tess_view: T)
  where
    T: Into<TessView<'b, C::Backend>>,
  {
    let tess_view = tess_view.into();

    unsafe {
      self.ctx.backend().render(
        &tess_view.tess.repr,
        tess_view.start_index,
        tess_view.vert_nb,
        tess_view.inst_nb,
      )
    }
  }
}
