use gl::types::*;

use crate::depth_test::DepthComparison;

pub(crate) fn depth_comparison_to_glenum(dc: DepthComparison) -> GLenum {
  match dc {
    DepthComparison::Never => gl::NEVER,
    DepthComparison::Always => gl::ALWAYS,
    DepthComparison::Equal => gl::EQUAL,
    DepthComparison::NotEqual => gl::NOTEQUAL,
    DepthComparison::Less => gl::LESS,
    DepthComparison::LessOrEqual => gl::LEQUAL,
    DepthComparison::Greater => gl::GREATER,
    DepthComparison::GreaterOrEqual => gl::GEQUAL,
  }
}
