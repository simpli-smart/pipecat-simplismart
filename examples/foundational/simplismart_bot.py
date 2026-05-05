# Copyright (c) 2026 Simplismart
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Voice agent using Simplismart STT / LLM / TTS — same layout as Pipecat
# ``examples/getting-started/06-voice-agent.py``:
# https://github.com/pipecat-ai/pipecat/blob/main/examples/getting-started/06-voice-agent.py
#
# Run with the Pipecat development runner (playground UI for WebRTC):
#   python simplismart_bot.py -t webrtc
# Then open http://localhost:7860/client

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat_simplismart import (
    SimplismartHttpTTSService,
    SimplismartLLMService,
    SimplismartSTTService,
)

load_dotenv(override=True)

# WebRTC (Pipecat runner playground at /client). To use Daily or Twilio, install
# ``pipecat-ai[daily]`` / telephony extras and extend this dict — see
# ``examples/getting-started/06-voice-agent.py``.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=SimplismartHttpTTSService.SAMPLE_RATE,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Simplismart voice bot")

    api_key = os.environ.get("SIMPLISMART_API_KEY")
    if not api_key:
        raise RuntimeError("Set SIMPLISMART_API_KEY in the environment or .env file.")

    stt = SimplismartSTTService(
        api_key=api_key,
        base_url=os.environ.get("SIMPLISMART_STT_URL", "https://api.simplismart.live/predict"),
    )

    tts = SimplismartHttpTTSService(
        api_key=api_key,
        base_url=os.environ.get("SIMPLISMART_TTS_URL", "https://api.simplismart.live/tts"),
        sample_rate=SimplismartHttpTTSService.SAMPLE_RATE,
    )

    llm = SimplismartLLMService(
        api_key=api_key,
        base_url=os.environ.get("SIMPLISMART_LLM_URL", "https://api.simplismart.live"),
        settings=SimplismartLLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses "
                "will be spoken aloud, so avoid emojis, bullet points, or other formatting "
                "that can't be spoken. Respond in a brief, natural way."
            ),
        ),
    )

    vad = SileroVADAnalyzer()
    vad_processor = VADProcessor(vad_analyzer=vad)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad_processor,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Please introduce yourself briefly to the user.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point for ``python simplismart_bot.py -t webrtc`` (and other transports)."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
