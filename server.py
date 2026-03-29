from fastapi import FastAPI, APIRouter, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import uuid
import base64
import tempfile
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class CertRequest(BaseModel):
    data: str       # base64 encoded file
    mimeType: str   # application/pdf, image/jpeg, etc.


@api_router.get("/")
async def root():
    return {"message": "AARS Karza API running"}


@api_router.post("/parse-cert")
async def parse_cert(req: CertRequest):
    """Proxy certificate parsing to AI via emergentintegrations."""
    temp_path = None
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

        file_bytes = base64.b64decode(req.data)
        ext_map = {
            "application/pdf": ".pdf",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        ext = ext_map.get(req.mimeType, ".pdf")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(file_bytes)
            temp_path = f.name

        api_key = os.environ.get("EMERGENT_LLM_KEY", "")
        chat = LlmChat(
            api_key=api_key,
            session_id="cert-" + str(uuid.uuid4()),
            system_message="You are a GST certificate data extractor. Return ONLY valid JSON.",
        ).with_model("gemini", "gemini-2.0-flash")

        file_content = FileContentWithMimeType(file_path=temp_path, mime_type=req.mimeType)
        prompt = (
            'Extract all GST certificate fields from this document. '
            'Return ONLY valid JSON with no markdown:\n'
            '{"gstin":"","legalName":"","tradeName":"","pan":"","state":"",'
            '"registrationDate":"","gstStatus":"","taxpayerType":"","constitution":"",'
            '"principalAddress":"","natureOfBusiness":[],"dateOfLiability":""}'
        )
        response = await chat.send_message(UserMessage(text=prompt, file_contents=[file_content]))

        text = str(response).strip()
        text = re.sub(r'```json|```', '', text).strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return json.loads(m.group(0))
        return {"error": "Could not parse certificate data"}

    except Exception as e:
        logger.error(f"Certificate parsing error: {e}")
        return {"error": str(e)}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
