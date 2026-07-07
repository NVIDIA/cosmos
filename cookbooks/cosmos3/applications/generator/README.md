# Generator Applications

Community and external applications using the Cosmos 3 Generator tower for
image, video, audio, action, and transfer generation.

## Structure

Generator applications are organized by capability, mirroring the core
cookbooks:

| Capability | Directory | Use cases |
|------------|-----------|-----------|
| [Audiovisual](audiovisual/) | `audiovisual/` | Text-to-image, text-to-video, image-to-video, audio generation |
| [Action](action/) | `action/` | Robotics policy, forward/inverse dynamics |
| [Transfer](transfer/) | `transfer/` | Video-to-video transfer with edge, depth, seg, blur, WSM controls |

## Environment Setup

All applications use the shared backend setup from
[`cookbooks/cosmos3/README.md`](../../README.md). Follow the section for the
backend your application uses (Cosmos Framework, Diffusers, or vLLM-Omni).

## Contributing

To add a new Generator application, create a directory under the appropriate
capability folder following the [Contributing Guide](../../../../CONTRIBUTING.md).
