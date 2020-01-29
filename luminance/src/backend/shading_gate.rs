use crate::backend::shader::Shader as ShaderBackend;

pub unsafe trait ShadingGate: ShaderBackend {
  unsafe fn apply_shader_program(&mut self, shader: &Self::ProgramRepr);
}
