# Copyright (c) 2026 Simplismart
#
# SPDX-License-Identifier: BSD-2-Clause
#
# Voice agent powered entirely by Simplismart — STT, LLM, and TTS.
#
# Follows the same structure as Pipecat's getting-started voice agent:
# https://github.com/pipecat-ai/pipecat/blob/main/examples/getting-started/06-voice-agent.py
#
# Usage:
#   python simplismart_bot.py -t webrtc
#   Then open http://localhost:7860/client

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
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
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_simplismart import SimplismartHttpTTSService, SimplismartLLMService, SimplismartSTTService

load_dotenv(override=True)

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. "
    "Keep your answers short and conversational — two or three sentences at most. "
    "Avoid markdown, bullet points, and any formatting that doesn't read naturally aloud."
)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=SimplismartHttpTTSService.SAMPLE_RATE,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    api_key = os.environ.get("SIMPLISMART_API_KEY")
    if not api_key:
        raise RuntimeError("SIMPLISMART_API_KEY is not set. Copy env.example to .env and fill it in.")

    stt = SimplismartSTTService(api_key=api_key)

    llm = SimplismartLLMService(
        api_key=api_key,
        settings=SimplismartLLMService.Settings(system_instruction=SYSTEM_PROMPT),
    )

    tts = SimplismartHttpTTSService(
        api_key=api_key,
        sample_rate=SimplismartHttpTTSService.SAMPLE_RATE,
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # SpeechTimeoutUserTurnStopStrategy fires the LLM as soon as the
            # STT transcript arrives (finalized=True) and the speech silence
            # window has passed — no fixed delay, no local ML model required.
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy()],
            ),
            # Safety net: if STT never returns a transcript (e.g. empty segment)
            # fall back to this timeout so the pipeline doesn't hang.
            user_turn_stop_timeout=3.0,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Hello"}],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
