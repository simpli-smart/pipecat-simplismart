# Simplismart.ai Plugin for Pipecat

Official [Pipecat](https://github.com/pipecat-ai/pipecat) integration for [Simplismart.ai](https://simplismart.ai): speech-to-text, streaming text-to-speech, and OpenAI-compatible chat completions—built for real-time multimodal AI pipelines.

**Maintained by Simplismart.**

**Tested with Pipecat v1.1.0** (`pipecat-ai>=0.0.86`).

## Install

Install from PyPI:

```bash
pip install pipecat-simplismart
```

Or with uv:

```bash
uv add pipecat-simplismart
```

The WebRTC quickstart below needs the **`example`** extra (Pipecat runner, Silero VAD, and WebRTC stack):

```bash
pip install "pipecat-simplismart[example]"
# or
uv add "pipecat-simplismart[example]"
```

**From source**:

```bash
git clone https://github.com/simplismart/pipecat-simplismart.git
cd pipecat-simplismart
uv pip install -e ".[example]"
```

## Usage

Each service follows Pipecat’s usual patterns (settings dataclass, frames, metrics).

### LLM (OpenAI-compatible)

```python
from pipecat_simplismart import SimplismartLLMService

llm = SimplismartLLMService(
    api_key="YOUR_KEY",
    base_url="https://api.simplismart.live",
    settings=SimplismartLLMService.Settings(model="google/gemma-4-31B-it"),
)
```

### STT (segmented / HTTP)

Place a ``VADProcessor`` **before** ``SimplismartSTTService`` so VAD events reach the segmented STT layer.

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat_simplismart import SimplismartSTTService

vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())
stt = SimplismartSTTService(
    api_key="YOUR_KEY",
    base_url="https://api.simplismart.live/predict",
)
# pipeline: ... transport.input(), vad_processor, stt, ...
```

### TTS

```python
from pipecat_simplismart import SimplismartHttpTTSService

tts = SimplismartHttpTTSService(
    api_key="YOUR_KEY",
    base_url="https://api.simplismart.live/tts",
    sample_rate=SimplismartHttpTTSService.SAMPLE_RATE,
    settings=SimplismartHttpTTSService.Settings(
        model="canopylabs/orpheus-3b-0.1-ft",
        voice="tara",
    ),
)
```

## Quickstart example

The example lives in this repository’s `examples/foundational/` directory (not in the PyPI wheel). Clone the repo, then run a small WebRTC bot: Silero VAD, this package’s STT/LLM/TTS, and the Pipecat runner CLI.

1. Copy `examples/foundational/env.example` to `examples/foundational/.env` and set `SIMPLISMART_API_KEY`.
2. With the **`example`** extra installed (see **Install**), from the clone root:

```bash
cd examples/foundational
python agent.py -t webrtc
```

If you use **uv** in a git checkout, from the repository root:

```bash
uv sync --extra example
cd examples/foundational
uv run --extra example python agent.py -t webrtc
```

3. Open the **Pipecat playground** at [http://localhost:7860/client](http://localhost:7860/client), connect, and speak.

## Environment variables

| Variable | Purpose |
|----------|---------|
| ``SIMPLISMART_API_KEY`` | Bearer JWT used for STT, TTS, and LLM |
| ``SIMPLISMART_STT_URL`` | Full URL for STT (default ``https://api.simplismart.live/predict``) |
| ``SIMPLISMART_TTS_URL`` | TTS endpoint (default ``https://api.simplismart.live/tts``) |
| ``SIMPLISMART_LLM_URL`` | OpenAI-compatible base URL (default ``https://api.simplismart.live``) |

## License

BSD 2-Clause — see [LICENSE](LICENSE).
