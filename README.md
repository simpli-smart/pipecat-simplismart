# pipecat-simplismart

Community-maintained [Pipecat](https://github.com/pipecat-ai/pipecat) integration for [Simplismart](https://simplismart.ai) — HTTP speech-to-text, streaming HTTP text-to-speech, and OpenAI-compatible chat completions.

**Maintained by Simplismart.**

**Tested with Pipecat v1.1.0** (``pipecat-ai>=0.0.86``).

## Install

After the package is published to PyPI:

```bash
pip install pipecat-simplismart
# or
uv add pipecat-simplismart
```

From a Git checkout until then:

```bash
uv pip install -e ".[example]"
```

## Usage (Pipecat pipeline)

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

### TTS (HTTP streaming)

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

## Foundational example (Pipecat runner + playground)

This mirrors the [06-voice-agent](https://github.com/pipecat-ai/pipecat/blob/main/examples/getting-started/06-voice-agent.py) layout: ``create_transport``, ``run_bot``, and ``main()`` from the Pipecat development runner.

1. Copy ``examples/foundational/env.example`` to ``examples/foundational/.env`` and set ``SIMPLISMART_API_KEY``.
2. From the ``pipecat-simplismart`` package directory:

```bash
uv pip install -e ".[example]"
cd examples/foundational
python simplismart_bot.py -t webrtc
```

3. Open the **Pipecat playground** at [http://localhost:7860/client](http://localhost:7860/client), connect, and speak.

If you use **uv** from the repo (e.g. ``uv run simplismart_bot.py``), install the example extras once so the runner and WebRTC UI are available, then pass the extra when running:

```bash
cd pipecat-simplismart
uv sync --extra example
cd examples/foundational
uv run --extra example python simplismart_bot.py -t webrtc
```

The ``[example]`` extra installs ``pipecat-ai[runner,webrtc,silero]`` (runner FastAPI server, Silero VAD, WebRTC, and the prebuilt Small WebRTC client UI).

> **Demo video:** add a short screen recording link here for your [Pipecat docs community integrations PR](https://github.com/pipecat-ai/docs).

## Environment variables

| Variable | Purpose |
|----------|---------|
| ``SIMPLISMART_API_KEY`` | Bearer JWT used for STT, TTS, and LLM |
| ``SIMPLISMART_STT_URL`` | Full URL for STT (default ``https://api.simplismart.live/predict``) |
| ``SIMPLISMART_TTS_URL`` | TTS endpoint (default ``https://api.simplismart.live/tts``) |
| ``SIMPLISMART_LLM_URL`` | OpenAI-compatible base URL (default ``https://api.simplismart.live``) |

## Listing in Pipecat docs

To appear on the [community integrations](https://docs.pipecat.ai/server/services/community-integrations) page, open a PR against [pipecat-ai/docs](https://github.com/pipecat-ai/docs) and post in **#community-integrations** on the Pipecat Discord.

## License

BSD 2-Clause — see [LICENSE](LICENSE).
