//! OpenGL backend.

pub mod buffer;
pub mod depth_test;
pub mod framebuffer;
pub mod pipeline;
pub mod pixel;
pub mod shader;
pub mod state;
pub mod tess;
pub mod texture;

use self::state::{GLState, StateQueryError};
use std::cell::RefCell;
use std::rc::Rc;

/// The OpenGL backend.
pub struct GL {
  pub(crate) state: Rc<RefCell<GLState>>,
}

impl GL {
  pub fn new() -> Result<Self, StateQueryError> {
    GLState::new().map(|state| GL {
      state: Rc::new(RefCell::new(state)),
    })
  }
}
