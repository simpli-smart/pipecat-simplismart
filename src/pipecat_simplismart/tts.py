"""HTTP streaming TTS via Simplismart ``/tts`` API."""

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TTSAudioRawFrame
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class SimplismartTTSSettings(TTSSettings):
    """Settings for SimplismartHttpTTSService.

    Parameters:
        temperature: Sampling temperature for speech generation.
        top_p: Top-p nucleus sampling.
        repetition_penalty: Repetition penalty.
        max_tokens: Max new tokens for the TTS model.
    """

    temperature: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    repetition_penalty: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_tokens: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SimplismartHttpTTSService(TTSService):
    """Streaming HTTP TTS; audio is raw PCM s16le mono at :attr:`SAMPLE_RATE`."""

    Settings = SimplismartTTSSettings
    _settings: Settings

    # Simplismart Orpheus-style models commonly use 24 kHz PCM output.
    SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Simplismart TTS.

        Args:
            api_key: Bearer token; defaults to ``SIMPLISMART_API_KEY``.
            base_url: TTS endpoint (e.g. ``https://api.simplismart.live/tts``).
            aiohttp_session: Optional shared aiohttp session.
            sample_rate: Output rate; should match :attr:`SAMPLE_RATE` for correct playback.
            settings: Voice, model, and generation parameters.
            **kwargs: Passed to :class:`TTSService`.
        """
        default_settings = self.Settings(
            model="canopylabs/orpheus-3b-0.1-ft",
            voice="tara",
            language=None,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            max_tokens=1000,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key or os.getenv("SIMPLISMART_API_KEY", "")
        if not self._api_key:
            raise ValueError("Simplismart TTS requires api_key or SIMPLISMART_API_KEY")

        self._base_url = base_url or os.getenv("SIMPLISMART_TTS_URL", "https://api.simplismart.live/tts")
        self._session: aiohttp.ClientSession | None = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings.output_sample_rate = self.sample_rate

    def _json_body(self, text: str) -> dict:
        return {
            "model": self._settings.model,
            "prompt": text,
            "voice": self._settings.voice,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "repetition_penalty": self._settings.repetition_penalty,
        }

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Simplismart TTS [{text[:80]}...]")
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            async with self._session.post(
                self._base_url,
                json=self._json_body(text),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    msg = f"Simplismart TTS HTTP {resp.status}: {err}"
                    await self.push_error(msg)
                    yield ErrorFrame(error=msg)
                    return

                async for chunk in resp.content.iter_chunked(1024):
                    if chunk:
                        yield TTSAudioRawFrame(
                            chunk,
                            self.sample_rate,
                            1,
                            context_id=context_id,
                        )

        except Exception as e:
            await self.push_error(f"Simplismart TTS error: {e}", exception=e)
            yield ErrorFrame(error=f"Simplismart TTS error: {e}")

    async def cleanup(self):
        await super().cleanup()
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None
