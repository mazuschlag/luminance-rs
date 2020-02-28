//! OpenGL 3.3 backend.

mod buffer;
mod depth_test;
mod framebuffer;
mod pipeline;
mod pixel;
mod shader;
mod state;
mod tess;
mod texture;

use self::state::GLState;
pub use self::state::StateQueryError;
use std::cell::RefCell;
use std::rc::Rc;

/// The OpenGL backend.
pub struct GL33 {
  pub(crate) state: Rc<RefCell<GLState>>,
}

impl GL33 {
  pub fn new() -> Result<Self, StateQueryError> {
    GLState::new().map(|state| GL33 {
      state: Rc::new(RefCell::new(state)),
    })
  }
}
