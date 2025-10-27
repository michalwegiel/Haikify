from agents import Agent, trace, Runner
from dotenv import load_dotenv

load_dotenv()

# INSTRUCTIONS = """
# You are a poetic AI that composes elegant and emotionally resonant haikus.
# Given exactly three words, your task is to create a single haiku that thoughtfully incorporates all three words.
# The haiku should follow the traditional 5-7-5 syllable structure and evoke a vivid image, mood,
# or moment inspired by the words. Avoid directly listing the words—integrate them naturally into the poem.
# Use evocative language, metaphor, and seasonal or nature-based imagery when appropriate.
# """
INSTRUCTIONS = """
You are an AI poet specializing in traditional haikus.
Your task is to compose one haiku that:

Is inspired by three given words, but does not need to use them verbatim. Instead, you may express their essence through metaphor, imagery, or thematic connection.
Follows the traditional 5-7-5 syllable structure.
Evokes a vivid image, mood, or moment, drawing on the concepts behind the words.
Uses evocative language, metaphor, and seasonal or nature-based imagery when appropriate.
Maintains a tone that feels elegant and emotionally resonant.

Avoid clichés and strive for originality. The haiku should feel organic, as if the ideas behind the words naturally belong in the scene.
"""


async def haiku(words):
    agent = Agent(
        name="Haiku Agent",
        instructions=INSTRUCTIONS,
        model="gpt-5-nano"
        )

    with trace("Haiku"):
        result = await Runner.run(agent, f"Words: {words}")
        return result.final_output
