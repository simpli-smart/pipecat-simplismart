"""Speech-to-text via Simplismart HTTP ``/predict`` API (segmented audio)."""

import base64
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def _wire_language(language: Language | str | None) -> str:
    """Map Pipecat language to a short code for the Simplismart API."""
    if language is None:
        return "en"
    s = language if isinstance(language, str) else str(language)
    return s.split("-")[0].lower()


@dataclass
class SimplismartSTTSettings(STTSettings):
    """Settings for Simplismart STT (extends common STT settings).

    Parameters:
        vad_filter: Enable server-side VAD filtering when supported.
        vad_onset: VAD onset threshold.
        beam_size: Beam search size for decoding.
        temperature: Decoding temperature.
    """

    vad_filter: bool = field(default=True)
    vad_onset: float = field(default=0.5)
    beam_size: int = field(default=4)
    temperature: float = field(default=0.0)


class SimplismartSTTService(SegmentedSTTService):
    """Segmented STT that POSTs WAV segments to Simplismart ``/predict``.

    Requires a :class:`~pipecat.processors.audio.vad_processor.VADProcessor` (or
    transport/user-aggregator VAD) upstream so ``VADUserStoppedSpeakingFrame`` is emitted.
    """

    Settings = SimplismartSTTSettings
    _settings: Settings

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
        """Initialize Simplismart STT.

        Args:
            api_key: Bearer token. Defaults to ``SIMPLISMART_API_KEY`` env var.
            base_url: Full URL to the predict endpoint (e.g. ``https://api.simplismart.live/predict``).
            aiohttp_session: Optional shared aiohttp session.
            sample_rate: Input audio sample rate; usually from pipeline ``StartFrame``.
            settings: STT settings including ``model`` and ``language``.
            **kwargs: Passed to :class:`SegmentedSTTService`.
        """
        default_settings = self.Settings(
            model="openai/whisper-large-v3-turbo",
            language=Language.EN,
            vad_filter=True,
            vad_onset=0.5,
            beam_size=4,
            temperature=0.0,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=sample_rate, settings=default_settings, **kwargs)

        self._api_key = api_key or os.getenv("SIMPLISMART_API_KEY", "")
        if not self._api_key:
            raise ValueError("Simplismart STT requires api_key or SIMPLISMART_API_KEY")

        self._base_url = base_url or os.getenv(
            "SIMPLISMART_STT_URL", "https://api.simplismart.live/predict"
        )
        self._session: aiohttp.ClientSession | None = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Whether this service participates in metrics collection."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Return wire-format language code for the API."""
        return _wire_language(language)

    def _payload(self, audio_b64: str) -> dict:
        lang = self._settings.language
        lang_str = _wire_language(lang) if lang is not None else "en"

        return {
            "model": self._settings.model,
            "audio_data": audio_b64,
            "language": lang_str,
            "task": "transcribe",
            "without_timestamps": True,
            "vad_model": "frame",
            "vad_filter": self._settings.vad_filter,
            "word_timestamps": False,
            "vad_onset": self._settings.vad_onset,
            "vad_offset": None,
            "min_speech_duration_ms": 0,
            "max_speech_duration_s": 30,
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400,
            "diarization": False,
            "initial_prompt": None,
            "hotwords": None,
            "num_speakers": 0,
            "compression_ratio_threshold": 2.4,
            "beam_size": self._settings.beam_size,
            "temperature": self._settings.temperature,
            "multilingual": False,
            "max_tokens": 400,
            "log_prob_threshold": -1.0,
            "length_penalty": 1,
            "repetition_penalty": 1.01,
            "strict_hallucination_reduction": False,
        }

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | str | None = None
    ):
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one WAV-encoded segment."""
        try:
            await self.start_processing_metrics()

            if not self._session:
                self._session = aiohttp.ClientSession()

            audio_b64 = base64.b64encode(audio).decode("utf-8")
            payload = self._payload(audio_b64)
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            async with self._session.post(
                self._base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    msg = f"Simplismart STT HTTP {resp.status}: {err_text}"
                    await self.push_error(msg)
                    yield ErrorFrame(error=msg)
                    return
                data = await resp.json()

        except Exception as e:
            await self.push_error(f"Simplismart STT error: {e}", exception=e)
            yield ErrorFrame(error=f"Simplismart STT error: {e}")
            return

        parts = data.get("transcription")
        if not parts:
            return
        text = "".join(parts).strip()
        if not text:
            return

        lang_enum = self._settings.language if isinstance(self._settings.language, Language) else None
        if lang_enum is None:
            lang_enum = Language.EN

        await self._handle_transcription(text, True, lang_enum)
        logger.debug(f"Simplismart transcription: [{text}]")

        yield TranscriptionFrame(
            text,
            self._user_id,
            time_now_iso8601(),
            lang_enum,
            result=data,
        )

    async def cleanup(self):
        await super().cleanup()
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None
