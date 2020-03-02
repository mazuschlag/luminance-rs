use gl::types::*;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::gl33::state::{BlendingState, DepthTest, FaceCullingState, GLState};
use crate::gl33::GL33;
use luminance::backend::pipeline::{
  Pipeline as PipelineBackend, PipelineBase, PipelineBuffer, PipelineTexture,
};
use luminance::backend::render_gate::RenderGate;
use luminance::backend::shading_gate::ShadingGate;
use luminance::backend::tess::Tess;
use luminance::backend::tess_gate::TessGate;
use luminance::pipeline::{PipelineError, PipelineState, Viewport};
use luminance::pixel::Pixel;
use luminance::render_state::RenderState;
use luminance::texture::Dimensionable;

#[non_exhaustive]
pub struct Pipeline {
  state: Rc<RefCell<GLState>>,
}

pub struct BoundBuffer {
  pub(crate) binding: u32,
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

pub struct BoundTexture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  pub(crate) unit: u32,
  state: Rc<RefCell<GLState>>,
  _phantom: PhantomData<*const (D, P)>,
}

impl<D, P> Drop for BoundTexture<D, P>
where
  D: Dimensionable,
  P: Pixel,
{
  fn drop(&mut self) {
    // place the binding into the free list
    let mut state = self.state.borrow_mut();
    state.binding_stack_mut().free_texture_units.push(self.unit);
  }
}

unsafe impl PipelineBase for GL33 {
  type PipelineRepr = Pipeline;

  unsafe fn new_pipeline(&mut self) -> Result<Self::PipelineRepr, PipelineError> {
    let pipeline = Pipeline {
      state: self.state.clone(),
    };

    Ok(pipeline)
  }
}

unsafe impl<D> PipelineBackend<D> for GL33
where
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

unsafe impl<T> PipelineBuffer<T> for GL33 {
  type BoundBufferRepr = BoundBuffer;

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
}

unsafe impl<D, P> PipelineTexture<D, P> for GL33
where
  D: Dimensionable,
  P: Pixel,
{
  type BoundTextureRepr = BoundTexture<D, P>;

  unsafe fn bind_texture(
    pipeline: &Self::PipelineRepr,
    texture: &Self::TextureRepr,
  ) -> Result<Self::BoundTextureRepr, PipelineError>
  where
    D: Dimensionable,
    P: Pixel,
  {
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
      _phantom: PhantomData,
    })
  }
}

unsafe impl TessGate for GL33 {
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

unsafe impl RenderGate for GL33 {
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

unsafe impl ShadingGate for GL33 {
  unsafe fn apply_shader_program(&mut self, shader_program: &Self::ProgramRepr) {
    self.state.borrow_mut().use_program(shader_program.handle);
  }
}
