# Changelog

## Unreleased

- Fix `ImportError` on pipecat-ai >= 1.8, which renamed `pipecat.services.settings._NotGiven` to `NotGiven`. The TTS module now imports whichever name the installed pipecat provides.

## 0.1.0 (2026-05-05)

- Initial community integration: `SimplismartSTTService`, `SimplismartHttpTTSService`, `SimplismartLLMService`.
- Example SmallWebRTC bot under `examples/foundational/`.
