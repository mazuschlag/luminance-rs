use crate::api::tess::TessView;
use crate::backend::tess::Tess as TessBackend;
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
  C::Backend: TessBackend,
{
  pub fn render<'b, T>(&'b mut self, tess: T)
  where
    T: Into<TessView<'b, C::Backend>>,
  {
    todo!("implement with backend::render + tess")
  }
}
