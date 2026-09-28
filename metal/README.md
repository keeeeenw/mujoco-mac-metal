# Metal package documentation

The [main project README](../README.md) covers current source and the **0.4.0** release, including
contributions, installation, all demo GIFs, measured performance and limitations.
This directory contains the optional experimental package and its technical guides.

- [Install and diagnose](INSTALL.md): dependencies, PyPI and source setup, GPU checks.
- [Supported profiles and qualification](DEVELOPMENT.md): precise physics coverage,
  CPU-reference evidence and remaining feature gaps.
- [API and state contracts](API.md): primitives, device buffers, stepping inputs,
  reset/restore and failure semantics.
- [Apple Silicon FAQ](FAQ.md): physics versus rendering, OpenGL, JAX, other backends
  and the scope of local validation.
- [Runnable demo index](examples/demo_gallery.md): capabilities and launch guides.
- [Benchmarks](benchmarks/README.md): CPU/Metal measurements and reproduction.
- [License](LICENSE) and [attribution](NOTICE).

Full MuJoCo compatibility remains unfinished. Use the selected profile's guards
and coverage guide when deciding whether a model is supported.
