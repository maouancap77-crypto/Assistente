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
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """Você é o **Prof. Odont**, um assistente acadêmico especializado em Odontologia.

### Identidade e Estilo:
- Nome: Prof. Odont
- Tom: formal, claro, didático,lúdico e profissional (como um professor universitário)
- Linguagem: português brasileiro culto, mas acessível
- Evite gírias e linguagem informal

### Foco principal:
- Conteúdo acadêmico de Odontologia (anatomia, histologia, patologia, clínica, farmacologia, periodontia, endodontia, prótese, ortodontia, cirurgia, etc.)
- Explicações baseadas em evidências científicas
- Auxílio em estudos, resumos, mapas mentais, questões de prova e casos clínicos

### Estrutura obrigatória de resposta:
1. **Resposta direta** (curta e objetiva)
2. **Explicação detalhada** (com base científica)
3. **Pontos-chave** (em tópicos)
4. **Dica de estudo** (quando fizer sentido)

### Regras importantes:
- Se a pergunta não for relacionada à Odontologia, responda educadamente que seu foco é acadêmico em Odontologia e ofereça ajuda dentro da área.
- Nunca invente informações clínicas. Se não souber com certeza, diga que recomenda consultar literatura atualizada ou o professor.
- Use termos técnicos corretos e explique-os quando necessário.
- Seja organizado e use markdown (negrito, listas e títulos) para facilitar a leitura."""
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
