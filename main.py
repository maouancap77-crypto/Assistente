import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=groq_api_key or "DUMMY_KEY",
    base_url="https://api.groq.com/openai/v1"
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"status": "Servidor da IA ativo!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not groq_api_key:
        raise HTTPException(
            status_code=500, 
            detail="A variável GROQ_API_KEY não foi configurada no Render."
        )
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",   # ← modelo atual recomendado
            messages=[
                {"role": "system", "content": "Você é um assistente virtual útil e objetivo."},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
