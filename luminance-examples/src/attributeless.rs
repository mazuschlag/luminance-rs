//! This program demonstrates how to render a triangle without sending anything to the GPU. This is
//! a not-so-famous technique to reduce the bandwidth and procedurally generate all the required
//! data to perform the render. The trick lives in ordering the GPU to render a certain number of
//! vertices and “spawn” the vertices’ data directly in the vertex shader by using the identifier of
//! the vertex currently being mapped over.
//!
//! Press <escape> to quit or close the window.
//!
//! https://docs.rs/luminance

use glfw::{Action, Context as _, Key, WindowEvent};
use luminance::context::GraphicsContext as _;
use luminance::pipeline::PipelineState;
use luminance::render_state::RenderState;
use luminance::shader::Program;
use luminance::tess::Mode;
use luminance::tess::TessBuilder;
use luminance_derive::{Semantics, Vertex};
use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};

const VS: &'static str = include_str!("attributeless-vs.glsl");
const FS: &'static str = include_str!("simple-fs.glsl");

fn main() {
  let mut surface = GlfwSurface::new_gl33(
    WindowDim::Windowed {
      width: 960,
      height: 540,
    },
    "Hello, world!",
    WindowOpt::default(),
  )
  .expect("GLFW surface creation");

  // we don’t use a Vertex type anymore (i.e. attributeless, so we use the unit () type)
  let mut program = Program::<_, (), (), ()>::from_strings(&mut surface, VS, None, None, FS)
    .expect("program creation")
    .ignore_warnings();

  // yet, we still need to tell luminance to render a certain number of vertices (even if we send no
  // attributes / data); in our case, we’ll just render a triangle, which has three vertices
  let tess = TessBuilder::new(&mut surface)
    .and_then(|b| b.set_vertex_nb(3))
    .and_then(|b| b.set_mode(Mode::Triangle))
    .and_then(|b| b.build())
    .unwrap();

  let mut back_buffer = surface.back_buffer().unwrap();
  let mut resize = false;

  'app: loop {
    surface.window.glfw.poll_events();
    for (_, event) in surface.events_rx.try_iter() {
      match event {
        WindowEvent::Close | WindowEvent::Key(Key::Escape, _, Action::Release, _) => break 'app,

        WindowEvent::FramebufferSize(..) => {
          resize = true;
        }

        _ => (),
      }
    }

    if resize {
      back_buffer = surface.back_buffer().unwrap();
      resize = true;
    }

    let render = surface.pipeline_gate().pipeline(
      &back_buffer,
      &PipelineState::default(),
      |_, mut shd_gate| {
        shd_gate.shade(&mut program, |_, mut rdr_gate| {
          rdr_gate.render(&RenderState::default(), |mut tess_gate| {
            // render the tessellation to the surface the regular way and let the vertex shader’s
            // magic do the rest!
            tess_gate.render(&tess);
          });
        });
      },
    );

    if render.is_ok() {
      surface.window.swap_buffers();
    } else {
      break 'app;
    }
  }
}
