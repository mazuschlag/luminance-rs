//! This program shows how to tweak the render state in order to render two simple triangles with
//! different parameters.
//!
//! From this tutorial on, vertex types and semantics are taken from a common.rs file.
//!
//! Press <space> to switch which triangle is rendered atop of which.
//! Press <b> to activate additive blending or disable it.
//! Press <escape> to quit or close the window.
//!
//! https://docs.rs/luminance

mod common;

use crate::common::{Semantics, Vertex, VertexColor, VertexPosition};
use glfw::{Action, Context as _, Key, WindowEvent};
use luminance::blending::{Equation, Factor};
use luminance::context::GraphicsContext as _;
use luminance::pipeline::PipelineState;
use luminance::render_state::RenderState;
use luminance::shader::Program;
use luminance::tess::{Mode, TessBuilder};
use luminance_glfw::GlfwSurface;
use luminance_windowing::{WindowDim, WindowOpt};

const VS: &'static str = include_str!("simple-vs.glsl");
const FS: &'static str = include_str!("simple-fs.glsl");

pub const TRI_RED_BLUE_VERTICES: [Vertex; 6] = [
  // first triangle – a red one
  Vertex {
    pos: VertexPosition::new([0.5, -0.5]),
    rgb: VertexColor::new([1., 0., 0.]),
  },
  Vertex {
    pos: VertexPosition::new([0.0, 0.5]),
    rgb: VertexColor::new([1., 0., 0.]),
  },
  Vertex {
    pos: VertexPosition::new([-0.5, -0.5]),
    rgb: VertexColor::new([1., 0., 0.]),
  },
  // second triangle, a blue one
  Vertex {
    pos: VertexPosition::new([-0.5, 0.5]),
    rgb: VertexColor::new([0., 0., 1.]),
  },
  Vertex {
    pos: VertexPosition::new([0.0, -0.5]),
    rgb: VertexColor::new([0., 0., 1.]),
  },
  Vertex {
    pos: VertexPosition::new([0.5, 0.5]),
    rgb: VertexColor::new([0., 0., 1.]),
  },
];

// Convenience type to demonstrate how the depth test influences the rendering of two triangles.
#[derive(Copy, Clone, Debug)]
enum DepthMethod {
  Under, // draw the red triangle under the blue one
  Atop,  // draw the red triangle atop the blue one
}

impl DepthMethod {
  fn toggle(self) -> Self {
    match self {
      DepthMethod::Under => DepthMethod::Atop,
      DepthMethod::Atop => DepthMethod::Under,
    }
  }
}

type Blending = Option<(Equation, Factor, Factor)>;

// toggle between no blending and additive blending
fn toggle_blending(blending: Blending) -> Blending {
  match blending {
    None => Some((Equation::Additive, Factor::One, Factor::One)),
    _ => None,
  }
}

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

  let mut program = Program::<_, Semantics, (), ()>::from_strings(&mut surface, VS, None, None, FS)
    .expect("program creation")
    .ignore_warnings();

  // create a red and blue triangles
  let red_triangle = TessBuilder::new(&mut surface)
    .and_then(|b| b.add_vertices(&TRI_RED_BLUE_VERTICES[0..3]))
    .and_then(|b| b.set_mode(Mode::Triangle))
    .and_then(|b| b.build())
    .unwrap();
  let blue_triangle = TessBuilder::new(&mut surface)
    .and_then(|b| b.add_vertices(&TRI_RED_BLUE_VERTICES[3..6]))
    .and_then(|b| b.set_mode(Mode::Triangle))
    .and_then(|b| b.build())
    .unwrap();

  let mut back_buffer = surface.back_buffer().unwrap();

  let mut blending = None;
  let mut depth_method = DepthMethod::Under;
  println!("now rendering red triangle {:?} the blue one", depth_method);

  let mut resize = false;

  'app: loop {
    surface.window.glfw.poll_events();
    for (_, event) in surface.events_rx.try_iter() {
      match event {
        WindowEvent::Close | WindowEvent::Key(Key::Escape, _, Action::Release, _) => break 'app,

        WindowEvent::Key(Key::Space, _, Action::Release, _) => {
          depth_method = depth_method.toggle();
          println!("now rendering red triangle {:?} the blue one", depth_method);
        }

        WindowEvent::Key(Key::B, _, Action::Release, _) => {
          blending = toggle_blending(blending);
          println!("now blending with {:?}", blending);
        }

        WindowEvent::FramebufferSize(..) => {
          resize = true;
        }

        _ => (),
      }
    }

    if resize {
      back_buffer = surface.back_buffer().unwrap();
      resize = false;
    }

    let render = surface.pipeline_gate().pipeline(
      &back_buffer,
      &PipelineState::default(),
      |_, mut shd_gate| {
        shd_gate.shade(&mut program, |_, _, mut rdr_gate| {
          let render_state = RenderState::default()
            // let’s disable the depth test so that every fragment (i.e. pixels) will be rendered to every
            // time we have to draw a part of a triangle
            .set_depth_test(None)
            // set the blending we decided earlier
            .set_blending(blending);

          rdr_gate.render(&render_state, |mut tess_gate| match depth_method {
            DepthMethod::Under => {
              tess_gate.render(&red_triangle);
              tess_gate.render(&blue_triangle);
            }

            DepthMethod::Atop => {
              tess_gate.render(&blue_triangle);
              tess_gate.render(&red_triangle);
            }
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
