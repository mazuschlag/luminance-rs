use crate::backend::tess::Tess;

pub unsafe trait TessGate: Tess {
  unsafe fn render(
    &mut self,
    tess: &Self::TessRepr,
    start_index: usize,
    vert_nb: usize,
    inst_nb: usize,
  );
}
