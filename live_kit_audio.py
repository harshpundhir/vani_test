import os
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm, ChatContext, ChatMessage
from livekit.plugins import silero,openai # silero, deepgram, cartesia, turn-detector
import asyncio
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pathlib import Path


load_dotenv()  # Load environment variables from parent folder .env file 



class VaniVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are Nora, a helpful voice AI assistant. Greet the user, listen for their question, and answer using the knowledge base.",
        )
        
    
    async def on_enter(self):
        await self.session.generate_reply(instructions="Greet the user and ask what they want to know.")

    # async def on_user_utterance(self, context, utterance: str):
    #     # Run the RAG pipeline with the user's question
    #     await self.session.generate_reply(instructions="Let me check that for you...")
    #     answer = await asyncio.to_thread(run_sitemap_pipeline, utterance)
    #     print(f"Answer: {answer}")
    #     await self.session.generate_reply(instructions=answer)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = VaniVoiceAgent()
    session = AgentSession(
        #turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        stt=openai.STT(
        model="gpt-4o-transcribe",
        ),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(
        model="gpt-4o-mini-tts",
        voice="ash",
        instructions="Speak in a friendly and conversational tone.",
        ),
    )
    await session.start(agent=agent, room=ctx.room)
    # The session will handle user input and responses

# For CLI/standalone use
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
