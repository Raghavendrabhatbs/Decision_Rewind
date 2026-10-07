from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend import process_podcast

app = FastAPI(
    title="AI Podcast Summarizer API",
    version="1.0.0"
)


@app.get("/")
def health():
    return {
        "status": "running",
        "message": "Podcast Summarizer API"
    }


@app.post("/summarize")
async def summarize_podcast(
    file: UploadFile = File(...),
    prompt: str = Form("")
):

    suffix = Path(file.filename).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        shutil.copyfileobj(file.file, temp_file)

        temp_path = temp_file.name

    try:

        result = process_podcast(
            temp_path,
            prompt
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        Path(temp_path).unlink(missing_ok=True)