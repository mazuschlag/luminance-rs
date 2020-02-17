use gl::types::*;

use crate::backend::gl::state::GLState;
use crate::backend::gl::GL;
use crate::backend::pipeline::{
  Pipeline as PipelineBackend, PipelineBase, PipelineError, PipelineState, Viewport,
};
use crate::backend::render_gate::RenderGate;
use crate::backend::shading_gate::ShadingGate;
use crate::backend::tess::Tess;
use crate::backend::tess_gate::TessGate;
use crate::backend::texture::{Dimensionable, Layerable};
use crate::blending::BlendingState;
use crate::depth_test::DepthTest;
use crate::face_culling::FaceCullingState;
use crate::render_state::RenderState;

use std::cell::RefCell;
use std::rc::Rc;

#[non_exhaustive]
pub struct Pipeline {
  state: Rc<RefCell<GLState>>,
}

pub struct BoundBuffer {
  binding: u32,
  state: Rc<RefCell<GLState>>,
}

impl Drop for BoundBuffer {
  fn drop(&mut self) {
    // place the binding into the free list
    let mut state = self.state.borrow_mut();
    state
      .binding_stack_mut()
      .free_buffer_bindings
      .push(self.binding);
  }
}

pub struct BoundTexture {
  unit: u32,
  state: Rc<RefCell<GLState>>,
}

impl Drop for BoundTexture {
  fn drop(&mut self) {
    // place the binding into the free list
    let mut state = self.state.borrow_mut();
    state.binding_stack_mut().free_texture_units.push(self.unit);
  }
}

unsafe impl PipelineBase for GL {
  type PipelineRepr = Pipeline;

  type BoundBufferRepr = BoundBuffer;

  type BoundTextureRepr = BoundTexture;

  unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError> {
    let pipeline = Pipeline {
      state: self.state.clone(),
    };

    Ok(pipeline)
  }

  unsafe fn bind_buffer(
    pipeline: &Self::PipelineRepr,
    buffer: &Self::BufferRepr,
  ) -> Result<Self::BoundBufferRepr, PipelineError> {
    let mut state = pipeline.state.borrow_mut();
    let bstack = state.binding_stack_mut();

    let binding = bstack.free_buffer_bindings.pop().unwrap_or_else(|| {
      // no more free bindings; reserve one
      let binding = bstack.next_buffer_binding;
      bstack.next_buffer_binding += 1;
      binding
    });

    state.bind_buffer_base(buffer.handle, binding);

    Ok(BoundBuffer {
      binding,
      state: pipeline.state.clone(),
    })
  }

  unsafe fn bind_texture(
    pipeline: &Self::PipelineRepr,
    texture: &Self::TextureRepr,
  ) -> Result<Self::BoundTextureRepr, PipelineError> {
    let mut state = pipeline.state.borrow_mut();
    let bstack = state.binding_stack_mut();

    let unit = bstack.free_texture_units.pop().unwrap_or_else(|| {
      // no more free units; reserve one
      let unit = bstack.next_texture_unit;
      bstack.next_texture_unit += 1;
      unit
    });

    state.set_texture_unit(unit);
    state.bind_texture(texture.target, texture.handle);

    Ok(BoundTexture {
      unit,
      state: pipeline.state.clone(),
    })
  }
}

unsafe impl<L, D> PipelineBackend<L, D> for GL
where
  L: Layerable,
  D: Dimensionable,
{
  unsafe fn start_pipeline(
    &mut self,
    framebuffer: &Self::FramebufferRepr,
    pipeline_state: &PipelineState,
  ) {
    let mut state = self.state.borrow_mut();

    state.bind_draw_framebuffer(framebuffer.handle);

    let PipelineState {
      clear_color,
      clear_color_enabled,
      clear_depth_enabled,
      viewport,
      srgb_enabled,
    } = *pipeline_state;
    let size = framebuffer.size;

    match viewport {
      Viewport::Whole => {
        state.set_viewport([0, 0, D::width(size) as GLint, D::height(size) as GLint]);
      }

      Viewport::Specific {
        x,
        y,
        width,
        height,
      } => {
        state.set_viewport([x as GLint, y as GLint, width as GLint, height as GLint]);
      }
    }

    state.set_clear_color([
      clear_color[0] as _,
      clear_color[1] as _,
      clear_color[2] as _,
      clear_color[3] as _,
    ]);

    if clear_color_enabled || clear_depth_enabled {
      let color_bit = if clear_color_enabled {
        gl::COLOR_BUFFER_BIT
      } else {
        0
      };
      let depth_bit = if clear_depth_enabled {
        gl::DEPTH_BUFFER_BIT
      } else {
        0
      };
      gl::Clear(color_bit | depth_bit);
    }

    state.enable_srgb_framebuffer(srgb_enabled);
  }
}

unsafe impl TessGate for GL {
  unsafe fn render(
    &mut self,
    tess: &Self::TessRepr,
    start_index: usize,
    vert_nb: usize,
    inst_nb: usize,
  ) {
    let _ = <Self as Tess>::render(tess, start_index, vert_nb, inst_nb);
  }
}

unsafe impl RenderGate for GL {
  unsafe fn enter_render_state(&mut self, rdr_st: &RenderState) {
    let mut gfx_state = self.state.borrow_mut();

    match rdr_st.blending {
      Some((equation, src_factor, dst_factor)) => {
        gfx_state.set_blending_state(BlendingState::On);
        gfx_state.set_blending_equation(equation);
        gfx_state.set_blending_func(src_factor, dst_factor);
      }
      None => {
        gfx_state.set_blending_state(BlendingState::Off);
      }
    }

    if let Some(depth_comparison) = rdr_st.depth_test {
      gfx_state.set_depth_test(DepthTest::On);
      gfx_state.set_depth_test_comparison(depth_comparison);
    } else {
      gfx_state.set_depth_test(DepthTest::Off);
    }

    match rdr_st.face_culling {
      Some(face_culling) => {
        gfx_state.set_face_culling_state(FaceCullingState::On);
        gfx_state.set_face_culling_order(face_culling.order);
        gfx_state.set_face_culling_mode(face_culling.mode);
      }
      None => {
        gfx_state.set_face_culling_state(FaceCullingState::Off);
      }
    }
  }
}

unsafe impl ShadingGate for GL {
  unsafe fn apply_shader_program(&mut self, shader_program: &Self::ProgramRepr) {
    self.state.borrow_mut().use_program(shader_program.handle);
  }
}
