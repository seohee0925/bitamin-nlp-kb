import os
import json
import time
from typing import Optional, Literal, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_PATH = os.path.join(BASE_DIR, "selected_cards.json")

from original_rag import FAISSRAGRetriever
from card_generator import CardGenerator

load_dotenv()
app = FastAPI(title="KB Card Dual RAG")

static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

last_recommendations: List[Dict[str, Any]] = []
selected_card: Optional[dict] = None
retriever = None
generator = CardGenerator()  # 전역 1회

def get_retriever():
    global retriever
    if retriever is None:
        from summary_rag import FAISSCardRetriever
        retriever = FAISSCardRetriever()
    return retriever

def save_selected_card(entry: dict):
    entry_to_save = {**entry, "timestamp": int(time.time())}
    try:
        if os.path.exists(SELECTED_PATH):
            with open(SELECTED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append(entry_to_save)
    with open(SELECTED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.get("/")
def home():
    filepath = os.path.join(BASE_DIR, "example.html")
    return FileResponse(
        filepath,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/healthz")
def healthz():
    return {"ok": True}

# === 추천 + 비교(비교 개수는 top_k와 동일) ===
@app.post("/recommend")
def recommend(
    user_input: str = Form(...),
    card_type: Literal["all", "credit", "check"] = Form("all"),
    top_k: int = Form(5),
):
    global last_recommendations
    r = get_retriever()
    items, _elapsed = r.find_similar_cards(user_input, card_type, top_k)
    last_recommendations = items or []

    # ⛔ 요약 말풍선 제거: summary_text 제공하지 않음(또는 빈 문자열)
    # summary_text = ""  # 필요하면 이렇게 명시적으로 빈 값

    # 비교는 항상 top_k 개수만큼
    comparison = generator.generate_comparison(last_recommendations, top_k=top_k)

    print(f"[SRV]/recommend items={len(last_recommendations)} top_k={top_k} "
          f"comparison_len={len(comparison or '')} head={(comparison or '')[:80]!r}")

    return JSONResponse({
        "items": last_recommendations,
        "summary_text": "",   # ← 프론트가 무시하도록 빈 문자열 전달
        "comparison": comparison
    })

@app.post("/select")
def select(
    card_name: str = Form(...),
    card_type: Literal["credit", "check"] = Form("credit"),
    keyword: str = Form(""),
):
    global selected_card
    picked = next((x for x in last_recommendations if x.get("card_name") == card_name), None)
    selected_card = {
        "card_name": card_name,
        "card_type": card_type,
        "keyword": keyword,
        "card_text": picked.get("card_text", "") if picked else "",
    }
    save_selected_card({"card_name": card_name, "card_type": card_type, "keyword": keyword, "from": "SummaryRAG"})
    return JSONResponse({"ok": True, "selected": selected_card})

@app.post("/rag")
def rag(
    question: str = Form(...),
    mode: Literal["detailed", "simple"] = Form("detailed"),
):
    if not selected_card:
        return JSONResponse({"message": "먼저 추천 목록에서 카드를 선택해 주세요."}, status_code=400)
    try:
        engine = FAISSRAGRetriever()
        explain_easy = mode == "simple"
        resp = engine.query(
            card_name=selected_card["card_name"],
            card_text=selected_card.get("card_text", ""),
            question=question,
            explain_easy=explain_easy,
        )
        if isinstance(resp, str):
            return JSONResponse({"card_name": selected_card["card_name"], "answer": resp, "sources": []})
        return JSONResponse({
            "card_name": selected_card["card_name"],
            "answer": resp.get("answer", ""),
            "sources": resp.get("sources", []),
        })
    except Exception as e:
        return JSONResponse({"message": f"오류: {e}"}, status_code=500)

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

@app.post("/simplify")
def simplify(text: str = Form(...)):
    try:
        msg = [
            {"role": "system", "content": "아래 글을 사용자가 정확하고 쉽게 이해할 수 있도록 재작성해."},
            {"role": "user", "content": text},
        ]
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=msg,
            temperature=0.3,
            max_tokens=600
        )
        return {"simplified": res.choices[0].message.content.strip()}
    except Exception as e:
        return JSONResponse({"message": f"간단화 오류: {e}"}, status_code=500)
