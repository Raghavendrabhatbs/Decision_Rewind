import os
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq
import json

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please configure it in the .env file."
    )

client = Groq(api_key=GROQ_API_KEY)

SUPPORTED_FORMATS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg"
}

def clean_json_response(content: str) -> str:
    """
    Removes Markdown code fences from an LLM response.
    """
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines)

    return content.strip()


def validate_audio(file_path: str) -> Path:
    """
    Validates the podcast audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        Path: Validated Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """

    audio_path = Path(file_path)

    # Check if file exists
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # Check if it is actually a file
    if not audio_path.is_file():
        raise ValueError(
            f"{audio_path} is not a valid file."
        )

    # Check supported extension
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{audio_path.suffix}'. "
            f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return audio_path

def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribes an audio file using the Groq Whisper API.

    Args:
        audio_path (Path): Validated audio file path.

    Returns:
        str: Transcript text.

    Raises:
        RuntimeError: If transcription fails.
    """

    try:
        with open(audio_path, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL,
                response_format="text"
            )

        transcript = transcription.strip()

        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript.")

        return transcript

    except Exception as e:
        raise RuntimeError(
            f"Failed to transcribe audio: {str(e)}"
        )

def build_prompt(transcript: str, user_prompt: str) -> str:
    """
    Builds the prompt for the Groq LLM.

    Args:
        transcript (str): Podcast transcript.
        user_prompt (str): User's custom instruction.

    Returns:
        str: Complete prompt.
    """

    if not user_prompt.strip():
        user_prompt = (
            "Generate a complete structured podcast analysis."
        )

    prompt = f"""
You are an expert podcast analyst.

Your task is to analyze the podcast transcript according to the user's request.

USER REQUEST:
{user_prompt}

If the user's request does not specify a custom format,
return the response as valid JSON with exactly the following schema:

{{
    "podcast_title": "",
    "summary": "",
    "five_key_points": [],
    "important_keywords": [],
    "main_topics": [],
    "action_items": [],
    "target_audience": "",
    "sentiment": "",
    "one_line_takeaway": ""
}}

Guidelines:

- Identify the podcast title if possible.
- Write a concise but informative summary.
- Extract exactly five key points.
- Extract meaningful keywords.
- Identify the major discussion topics.
- List actionable recommendations if present.
- Identify the intended audience.
- Determine the overall sentiment.
- Write a one-line takeaway.

Podcast Transcript:

{transcript}

Return ONLY valid JSON.
"""

    return prompt

def generate_summary(transcript: str, user_prompt: str) -> dict:
    """
    Generates a structured podcast summary using the Groq LLM.

    Args:
        transcript (str): Podcast transcript.
        user_prompt (str): User instruction.

    Returns:
        dict: Structured podcast analysis.
    """

    prompt = build_prompt(transcript, user_prompt)

    try:

        response = client.chat.completions.create(

            model=LLM_MODEL,

            temperature=TEMPERATURE,

            max_completion_tokens=MAX_TOKENS,

            messages=[
                {
                    "role": "system",
                    "content": "You are an expert podcast analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = clean_json_response(content)
        result = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(
            "The LLM returned an invalid JSON response."
        )

    except Exception as e:
        raise RuntimeError(
            f"Summary generation failed: {e}"
        )

def process_podcast(file_path: str, user_prompt: str = "") -> dict:
    """
    Complete podcast processing pipeline.

    Args:
        file_path (str): Path to the podcast audio file.
        user_prompt (str): Optional custom user instruction.

    Returns:
        dict: Structured podcast analysis.
    """

    # Step 1: Validate the audio file
    audio_path = validate_audio(file_path)

    # Step 2: Generate transcript
    transcript = transcribe_audio(audio_path)

    # Step 3: Generate structured summary
    analysis = generate_summary(transcript, user_prompt)

    # Step 4: Include transcript in the response
    analysis["transcript"] = transcript

    return analysis



if __name__ == "__main__":

    try:

        result = process_podcast(
            file_path="podcast.mp3",
            user_prompt="Summarize this podcast for college students."
        )

        print("\n===== PODCAST ANALYSIS =====\n")

        for key, value in result.items():

            print(f"{key}:\n")

            if isinstance(value, list):

                for item in value:
                    print(f"• {item}")

            else:
                print(value)

            print()

    except Exception as e:
        print(f"Error: {e}")
