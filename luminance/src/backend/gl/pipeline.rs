use gl::types::*;

use crate::backend::framebuffer::Framebuffer as FramebufferBackend;
use crate::backend::gl::framebuffer::Framebuffer;
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

#[non_exhaustive]
pub struct Pipeline;

unsafe impl PipelineBase for GL {
  type PipelineRepr = Pipeline;

  unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError> {
    Ok(Pipeline)
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
  ) where
    L: Layerable,
    D: Dimensionable,
  {
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
    <Self as Tess>::render(tess, start_index, vert_nb, inst_nb);
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
