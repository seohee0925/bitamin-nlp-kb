# original_rag.py  (혹은 사용 중인 파일명)
import os
import re
import json
import hashlib
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import TypedDict, Optional

import numpy as np
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage



_PUNCT = re.compile(r"[ \t\r\n\-_()/\[\]{}·.,!?'\"…]+")

def _normalize_name(s: str) -> str:
    """파일명/카드명을 비교하기 위한 정규화: 공백/구두점 제거 + 소문자."""
    if not s:
        return ""
    s = s.strip()
    # 확장자/자주 붙는 접미어 제거
    s = re.sub(r"\.json$", "", s, flags=re.IGNORECASE)
    for suf in ("_정제", "_최종", "_clean", "_final"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return _PUNCT.sub("", s).lower()


class GeneratorState(TypedDict):
    card_name: str
    user_question: str
    context_chunks: list[str]
    prompt: str
    answer: str
    simplified_answer: str
    explain_easy: bool


class FAISSRAGRetriever:
    """Original RAG 시스템 - 카드별 상세 정보 검색 및 질의응답"""

    def __init__(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        load_dotenv()

        # === 프로젝트 기준 경로 설정 ===
        proj: Path = Path(__file__).resolve().parent        # fastapi_project 폴더
        self.current_file_dir = str(proj)                   # 문자열 경로도 보관
        self.base_path = str(proj)                          # 과거 호환

        def pick_dir(label: str, candidates: list[Path]) -> Optional[str]:
            for p in candidates:
                if p.exists():
                    print(f"📂 {label} 디렉토리: {p}")
                    return str(p)
            tried = "\n  - " + "\n  - ".join(map(str, candidates))
            print(f"⚠️ {label} 디렉토리를 찾지 못했습니다. 시도한 경로:{tried}")
            return None

        # 신용/체크 카드 루트 디렉토리 자동 탐색 (우선순위 순)
        self.credit_dir = pick_dir(
            "신용카드",
            [
                proj / "신용카드",
                proj / "JSON" / "신용json",
                proj.parent / "신용카드",
            ],
        )
        self.check_dir = pick_dir(
            "체크카드",
            [
                proj / "체크카드",
                proj / "JSON" / "체크json",
                proj.parent / "체크카드",
            ],
        )

        # 이후 순회할 루트 목록
        self.data_dirs = [d for d in [self.credit_dir, self.check_dir] if d]
        if not self.data_dirs:
            raise FileNotFoundError("신용카드/체크카드 데이터 디렉토리를 찾지 못했습니다.")

        # selected_cards.json 위치
        self.selected_cards_path = os.path.join(self.current_file_dir, "selected_cards.json")

        # 임베딩 저장 디렉토리
        self.embeddings_dir = os.path.join(self.current_file_dir, "original_embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        print(f"📁 임베딩 저장 디렉토리: {self.embeddings_dir}")

        # 모델들
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
        )
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # LangGraph
        self.graph = self._build_langgraph()

        # 캐시
        self._document_cache: dict[str, list[Document]] = {}
        self._faiss_cache: dict[str, FAISS] = {}
        self._bm25_cache: dict[str, BM25Retriever] = {}

        print("🎉 FAISSRAGRetriever 초기화 완료!")

    # ----------------- 유틸/로딩 -----------------
    def get_latest_card_from_selected_cards(self) -> Optional[str]:
        try:
            if not os.path.exists(self.selected_cards_path):
                print(f"❌ selected_cards.json 파일을 찾을 수 없습니다: {self.selected_cards_path}")
                return None

            with open(self.selected_cards_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                print("❌ selected_cards.json이 비어있습니다")
                return None

            last_card = data[-1]
            card_name = last_card.get("card_name")
            print("🎯 마지막 카드 정보:")
            print(f"   - Card Name: {card_name}")
            print(f"   - Timestamp: {last_card.get('timestamp', 'N/A')}")
            print(f"   - Question: {last_card.get('question', 'N/A')}")
            print(f"   - Card Type: {last_card.get('card_type', 'N/A')}")
            print(f"   - Keyword: {last_card.get('keyword', 'N/A')}")
            return card_name
        except Exception as e:
            print(f"❌ selected_cards.json 읽기 중 오류: {e}")
            return None

    def load_documents_field_level(self, json_paths: list[str]) -> list[Document]:
        print(f"📋 문서 로딩 시작... (총 {len(json_paths)}개 파일)")
        documents: list[Document] = []

        for json_path in tqdm(json_paths, desc="📋 문서 로딩 중"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                card_name = data.get("card_name", "UnknownCard")

                for section in data.get("sections", []):
                    heading = section.get("heading", "")
                    subheading = section.get("subheading", "")

                    for key in [
                        "benefit",
                        "benefits",
                        "fee",
                        "agreement",
                        "condition",
                        "conditions",
                        "etc",
                        "overseas_usage",
                    ]:
                        val = section.get(key)
                        if not val:
                            continue
                        contents = val if isinstance(val, list) else [val]
                        merged_text = "\n".join([c.strip() for c in contents if len(c.strip()) > 0])
                        if len(merged_text.strip()) < 10:
                            continue

                        full_text = f"[{card_name}]\n{heading} - {subheading}\n<{key}>\n{merged_text}"
                        documents.append(
                            Document(
                                page_content=full_text,
                                metadata={
                                    "card_name": card_name,
                                    "field": key,
                                    "heading": heading,
                                    "subheading": subheading,
                                },
                            )
                        )
            except Exception as e:
                print(f"❗ Error reading {json_path}: {e}")

        return documents

    def _get_card_category_from_path(self, json_path: str) -> str:
        p = json_path.lower()
        # 한국어 폴더명과 영문 키워드 모두 대응
        if ("신용" in p) or ("credit" in p):
            return "credit"
        if ("체크" in p) or ("check" in p):
            return "check"
        return "unknown"

    def _load_category_documents(self, category: str) -> list[Document]:
        """카테고리별 모든 문서 로딩 (하위폴더 재귀 탐색)"""
        print(f"📋 {category.upper()} 카드 카테고리 문서 로딩 중...")

        root_dir = self.credit_dir if category == "credit" else self.check_dir if category == "check" else None
        if not root_dir or not os.path.exists(root_dir):
            print(f"❌ 카테고리 디렉토리를 찾을 수 없습니다: {root_dir}")
            return []

        json_files = [str(p) for p in Path(root_dir).rglob("*.json")]
        print(f"📊 {category.upper()} 카테고리에서 {len(json_files)}개 JSON 파일 발견")

        if not json_files:
            print(f"⚠️ {category.upper()} 카테고리에 JSON 파일이 없습니다")
            return []

        return self.load_documents_field_level(json_files)

    def _save_category_embeddings(self, category: str, documents: list[Document], faiss_index: FAISS) -> bool:
        print(f"💾 {category.upper()} 카드 임베딩 저장 중...")

        try:
            base_filename = f"{category}_card"
            faiss_path = os.path.join(self.embeddings_dir, f"{base_filename}_embeddings.faiss")
            pkl_path = os.path.join(self.embeddings_dir, f"{base_filename}_embedding_data.pkl")
            metadata_path = os.path.join(self.embeddings_dir, f"{base_filename}_metadata.json")
            texts_path = os.path.join(self.embeddings_dir, f"{base_filename}_texts.txt")

            print("   📁 저장 경로:")
            print(f"      - FAISS 인덱스: {faiss_path}")
            print(f"      - 임베딩 데이터: {pkl_path}")
            print(f"      - 메타데이터: {metadata_path}")
            print(f"      - 텍스트: {texts_path}")

            # 1) faiss raw index 저장
            if hasattr(faiss_index, "index"):
                import faiss as faiss_lib

                faiss_lib.write_index(faiss_index.index, faiss_path)

            # 2) 문서/텍스트/메타 저장
            embedding_data = {
                "documents": documents,
                "texts": [doc.page_content for doc in documents],
                "metadatas": [doc.metadata for doc in documents],
                "embeddings": (
                    faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal)
                    if hasattr(faiss_index.index, "reconstruct_n")
                    else None
                ),
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(embedding_data, f)

            import datetime

            metadata_summary = {
                "category": category,
                "total_documents": len(documents),
                "cards": list({doc.metadata.get("card_name", "Unknown") for doc in documents}),
                "created_at": str(datetime.datetime.now().isoformat()),
                "embedding_model": "BAAI/bge-m3",
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_summary, f, ensure_ascii=False, indent=2)

            with open(texts_path, "w", encoding="utf-8") as f:
                f.write(f"=== {category.upper()} 카드 문서 텍스트 ===\n")
                f.write(f"총 문서 수: {len(documents)}\n")
                f.write(f"포함된 카드: {', '.join(metadata_summary['cards'])}\n")
                f.write("=" * 50 + "\n\n")
                for i, doc in enumerate(documents):
                    f.write(f"문서 {i+1}:\n")
                    f.write(f"카드명: {doc.metadata.get('card_name', 'Unknown')}\n")
                    f.write(f"필드: {doc.metadata.get('field', 'Unknown')}\n")
                    f.write(f"제목: {doc.metadata.get('heading', '')} - {doc.metadata.get('subheading', '')}\n")
                    f.write("-" * 30 + "\n")
                    body = doc.page_content
                    f.write(body[:500] + "...\n" if len(body) > 500 else body + "\n")
                    f.write("\n" + "=" * 50 + "\n\n")

            print(f"✅ {category.upper()} 카드 임베딩 저장 성공!")
            return True
        except Exception as e:
            print(f"❌ {category.upper()} 카드 임베딩 저장 실패: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _load_category_embeddings(self, category: str) -> tuple[Optional[list[Document]], Optional[FAISS], Optional[BM25Retriever]]:
        base_filename = f"{category}_card"
        faiss_path = os.path.join(self.embeddings_dir, f"{base_filename}_embeddings.faiss")
        pkl_path = os.path.join(self.embeddings_dir, f"{base_filename}_embedding_data.pkl")

        if not os.path.exists(pkl_path):
            print(f"⚠️ {category.upper()} 카드 임베딩 파일이 없습니다: {pkl_path}")
            return None, None, None

        try:
            with open(pkl_path, "rb") as f:
                embedding_data = pickle.load(f)
            documents: list[Document] = embedding_data["documents"]

            if os.path.exists(faiss_path):
                import faiss as faiss_lib

                index = faiss_lib.read_index(faiss_path)
                faiss_index = FAISS(
                    embedding_function=self.embedding_model,
                    index=index,
                    docstore={i: doc for i, doc in enumerate(documents)},
                    index_to_docstore_id={i: i for i in range(len(documents))},
                )
                print("   🔍 FAISS 인덱스 로드 완료")
            else:
                print("   ⚠️ FAISS 파일이 없어 재생성합니다...")
                faiss_index = FAISS.from_documents(documents, self.embedding_model)

            bm25 = BM25Retriever.from_documents(documents)
            bm25.k = 60

            print(f"✅ {category.upper()} 카드 임베딩 로드 성공!")
            return documents, faiss_index, bm25
        except Exception as e:
            print(f"❌ {category.upper()} 카드 임베딩 로드 실패: {e}")
            import traceback

            traceback.print_exc()
            return None, None, None

    def build_category_embeddings(self, force_rebuild: bool = False):
        for category in ["credit", "check"]:
            print("\n" + "=" * 50)
            base_filename = f"{category}_card"
            pkl_path = os.path.join(self.embeddings_dir, f"{base_filename}_embedding_data.pkl")

            if os.path.exists(pkl_path) and not force_rebuild:
                print(f"✅ {category.upper()} 카드 임베딩이 이미 존재합니다: {pkl_path}")
                print("   force_rebuild=True로 설정하면 재빌드됩니다.")
                continue

            documents = self._load_category_documents(category)
            if not documents:
                print(f"⚠️ {category.upper()} 카테고리에 문서가 없습니다. 스킵합니다.")
                continue

            cards = list({doc.metadata.get("card_name", "Unknown") for doc in documents})
            print("📊 통계:")
            print(f"   - 총 문서 수: {len(documents)}")
            print(f"   - 카드 수: {len(cards)}")
            print(f"   - 샘플: {', '.join(cards[:5])}{'...' if len(cards) > 5 else ''}")

            print(f"🔍 {category.upper()} 카드 FAISS 인덱스 생성 중...")
            faiss_index = FAISS.from_documents(documents, self.embedding_model)
            print("✅ 생성 완료")

            if self._save_category_embeddings(category, documents, faiss_index):
                print(f"🎉 {category.upper()} 카드 임베딩 빌드 완료!")
            else:
                print(f"❌ {category.upper()} 카드 임베딩 빌드 실패!")

        print(f"📁 저장 위치: {self.embeddings_dir}")

    # ----------------- 검색/생성 -----------------
    def reciprocal_rank_fusion(self, faiss_results: list, bm25_results: list, k: int = 60) -> list[str]:
        scores = defaultdict(float)

        def update_scores(results, weight):
            for rank, item in enumerate(results):
                key = item.page_content.strip()
                scores[key] += weight / (k + rank + 1)

        update_scores(faiss_results, weight=0.6)
        update_scores(bm25_results, weight=0.4)

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in sorted_chunks[:k]]

    def build_generator_prompt(self, card_name: str, user_question: str, context_chunks: list[str]) -> str:
        context_text = "\n".join(context_chunks)
        return f"""
당신은 신용카드 및 체크카드 정보를 바탕으로 사용자 질문에 대해 **문서 기반으로 구체적이고 정확한 답변을 제공하는 AI 전문가**입니다.

카드 이름: {card_name}

아래는 해당 카드의 약관 및 상품설명서에서 추출된 일부 문서 내용입니다:

[문서 내용]
{context_text}

[사용자 질문]
{user_question}

[응답 조건]
1. 반드시 **사용자의 질문에 맞게 위 문서 내용에 기반**하여 답변하고, 가능한 한 **구체적이고 풍부한 설명**을 제공하세요.
   (예: 혜택 업종, 혜택 조건, 부가 혜택, 해지 조건 등)
2. 사용자의 질문에 답변할 때 **정확한 수치, 시기, 조건, 문구**를 반영하세요.
   (예: "월 최대 2회", "국내전용 9,000원", "전월 30만원 이상 이용 시 적용")
3. **카드 약관의 주의사항이나 확인사항** (예: 전월 실적 제외 조건, 소비자 권리, 연체 시 불이익 등)도 문맥에 따라 포함해 주세요.
4. 사용자의 질문이 여러 항목을 포함할 경우, 각 항목별로 **체계적으로 구성하여 답변**하세요.
5. 문서에 명시되지 않은 내용은 "문서에 명시된 정보가 없습니다."라고 답변하고, **절대 거짓 정보를 지어내서는 안됩니다.**
""".strip()

    def _build_prompt_node(self, state: GeneratorState) -> GeneratorState:
        state["prompt"] = self.build_generator_prompt(
            card_name=state["card_name"],
            user_question=state["user_question"],
            context_chunks=state["context_chunks"],
        )
        return state

    def _generate_answer_node(self, state: GeneratorState) -> GeneratorState:
        messages = [
            SystemMessage(content="당신은 카드별 정보 기반 응답을 정확하게 생성하는 전문가입니다."),
            HumanMessage(content=state["prompt"]),
        ]
        state["answer"] = self.llm(messages).content
        return state

    def _rewrite_answer_node(self, state: GeneratorState) -> GeneratorState:
        if not state.get("explain_easy", False):
            state["simplified_answer"] = ""
            return state

        prompt = f"""
당신은 신용카드나 체크카드 정보를 사용자가 **정확하고 쉽게 이해할 수 있도록 재작성**해주는 AI입니다.

아래 문장을 다음 기준에 따라 **친절하게 다시 설명**해주세요:

1. **전문 용어**(예: 리볼빙, 위법계약해지권 등)는 간단한 예시나 쉬운 말로 풀어서 설명하세요.
2. **문장이 길고 복잡한 경우**, **핵심을 유지**하면서 문장을 분리해 **명확하게 정리**하세요.
3. **필수 정보**(예: 금액, 조건, 책임, 유의사항 등)는 **절대 빠뜨리지 말고 반영**하세요.
4. 요약하지 말고, 말을 지어내지 말고, 법적 표현을 삭제하지 말고 **문맥 그대로 쉽게 풀어** 쓰세요.
5. 전체적으로 **고객 상담원이 친절하게 설명해주는 말투**로 바꾸세요.

[원문]
{state['answer']}

[쉬운 설명]
"""
        messages = [
            SystemMessage(content="당신은 금융 정보를 쉽게 설명해주는 AI입니다."),
            HumanMessage(content=prompt),
        ]
        
        gpt4_llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        state["simplified_answer"] = gpt4_llm(messages).content
        return state

    def _build_langgraph(self) -> StateGraph:
        builder = StateGraph(GeneratorState)
        builder.add_node("BuildPrompt", self._build_prompt_node)
        builder.add_node("GenerateAnswer", self._generate_answer_node)
        builder.add_node("RewriteAnswer", self._rewrite_answer_node)
        builder.set_entry_point("BuildPrompt")
        builder.add_edge("BuildPrompt", "GenerateAnswer")
        builder.add_edge("GenerateAnswer", "RewriteAnswer")
        builder.add_edge("RewriteAnswer", END)
        return builder.compile()

    # -------- 카드 파일 찾기 --------
    def _find_card_json_path(self, card_name: str) -> Optional[str]:
        """카드 이름으로 JSON 파일 경로 찾기 (하위 폴더 재귀 + 부분 매칭)."""
        print(f"🔍 '{card_name}' 카드의 JSON 파일 검색 중...")
        needle = _normalize_name(card_name)

        # 후보 (score, path)로 모아 가장 점수 높은 파일 선택
        candidates: list[tuple[int, str]] = []
        total_checked = 0

        def score_match(target: str) -> int:
            """
            일치 점수 (높을수록 우선)
              100: 완전 동일
               80: 시작 부분 일치(또는 상호 prefix)
               60: 부분 포함
                0: 불일치
            """
            t = _normalize_name(target)
            if t == needle:
                return 100
            if t.startswith(needle) or needle.startswith(t):
                return 80
            if needle in t:
                return 60
            return 0

        for data_dir in self.data_dirs:
            if not data_dir or not os.path.exists(data_dir):
                continue

            for root, _, files in os.walk(data_dir):
                json_files = [f for f in files if f.lower().endswith(".json")]
                total_checked += len(json_files)

                for fn in json_files:
                    full = os.path.join(root, fn)

                    # 1) 파일명 기반 점수
                    s1 = score_match(fn)

                    # 2) JSON 내부 card_name 기반 점수
                    s2 = 0
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        inner = data.get("card_name") or ""
                        s2 = score_match(inner)
                    except Exception:
                        pass

                    score = max(s1, s2)
                    if score >= 60:  # 부분 포함 이상만 후보로 채택
                        # 파일명 완전 동일이면 소폭 가중치
                        if _normalize_name(fn) == needle:
                            score += 5
                        candidates.append((score, full))

        print(f"   🔎 검사한 JSON 파일 수: {total_checked}")
        if not candidates:
            print("   ❌ 매칭 후보가 없습니다.")
            self._show_available_cards_sample()
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = candidates[0]
        print(f"✅ 최적 매칭({best_score}): {best_path}")
        return best_path


    def _show_available_cards_sample(self, max_samples: int = 5):
        found_cards = []
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                continue
            for p in Path(data_dir).rglob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("card_name")
                    if name:
                        cat = os.path.relpath(str(p.parent), data_dir)
                        found_cards.append((name, cat))
                        print(f"   - '{name}' (카테고리: {cat})")
                        if len(found_cards) >= max_samples:
                            raise StopIteration
                except StopIteration:
                    break
                except Exception:
                    continue
            if len(found_cards) >= max_samples:
                break
        if not found_cards:
            print("   ❌ 사용 가능한 카드를 찾을 수 없습니다.")
        else:
            print(f"   💡 총 {len(found_cards)}개 샘플 표시 (실제로는 더 많을 수 있음)")

    # -------- 카드별 준비/검색 --------
    def _prepare_individual_card_data(self, card_name: str, json_path: str):
        documents = self.load_documents_field_level([json_path])
        if not documents:
            raise ValueError(f"'{card_name}' 카드의 문서가 비어 있습니다.")

        faiss_index = FAISS.from_documents(documents, self.embedding_model)
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 60
        return documents, faiss_index, bm25

    def _prepare_card_data(self, card_name: str):
        print(f"🔧 '{card_name}' 카드 데이터 준비 중...")

        if card_name in self._document_cache:
            print("💾 개별 카드 캐시 사용 중...")
            return (
                self._document_cache[card_name],
                self._faiss_cache[card_name],
                self._bm25_cache[card_name],
            )

        json_path = self._find_card_json_path(card_name)
        if not json_path:
            raise ValueError(f"'{card_name}' 카드의 데이터를 찾을 수 없습니다.")

        category = self._get_card_category_from_path(json_path)
        cat_docs, cat_faiss, cat_bm25 = self._load_category_embeddings(category)

        if cat_docs is None:
            print(f"⚠️ {category.upper()} 카테고리 임베딩이 없어 개별 처리합니다...")
            docs, fidx, bidx = self._prepare_individual_card_data(card_name, json_path)
        else:
            needle = _normalize_name(card_name)
            card_docs = [d for d in cat_docs if _normalize_name(d.metadata.get("card_name","")) == needle]
            if not card_docs:
                raise ValueError(f"카테고리 임베딩에 '{card_name}' 카드가 없습니다.")
            fidx = FAISS.from_documents(card_docs, self.embedding_model)
            bidx = BM25Retriever.from_documents(card_docs); bidx.k = 60
            docs = card_docs

        self._document_cache[card_name] = docs
        self._faiss_cache[card_name] = fidx
        self._bm25_cache[card_name] = bidx
        return docs, fidx, bidx

    # -------- 질의 --------
    def query(self, card_name: str, card_text: str, question: str, explain_easy: bool = False, top_k: int = 20) -> str:
        print("\n" + "=" * 60)
        print("🚀 질의응답 시작")
        print(f"   - 카드명: {card_name}")
        print(f"   - 질문: {question}")
        print(f"   - 쉬운 설명: {explain_easy}")
        print(f"   - Top-K: {top_k}")
        print("=" * 60)

        try:
            documents, faiss_index, bm25 = self._prepare_card_data(card_name)

            print("📊 문서 검색 및 랭킹 중...")
            faiss_results = faiss_index.similarity_search(question, k=60)
            bm25_results = bm25.get_relevant_documents(question)

            rrf_candidates = self.reciprocal_rank_fusion(faiss_results, bm25_results)
            print(f"🔄 Cross-Encoder 재랭킹 실행... (총 {len(rrf_candidates)}개 후보)")
            inputs = [(question, chunk) for chunk in rrf_candidates]
            scores = self.reranker.predict(inputs)
            reranked_chunks = [x for _, x in sorted(zip(scores, rrf_candidates), reverse=True)]
            top_chunks = reranked_chunks[:top_k]
            print(f"✅ 재랭킹 완료 - 최종 {len(top_chunks)}개 청크 선택")

            initial_state: GeneratorState = {
                "card_name": card_name,
                "user_question": question,
                "context_chunks": top_chunks,
                "prompt": "",
                "answer": "",
                "simplified_answer": "",
                "explain_easy": explain_easy,
            }
            result = self.graph.invoke(initial_state)

            final_answer = result["simplified_answer"] if (explain_easy and result["simplified_answer"]) else result["answer"]
            print(f"🎉 질의응답 완료! (최종 답변 길이: {len(final_answer)}자)")
            return final_answer
        except Exception as e:
            msg = f"❌ '{card_name}' 카드 질의응답 중 오류가 발생했습니다: {e}"
            print(msg)
            return msg

    # -------- 기타 --------
    def clear_cache(self):
        print("🗑️ 캐시 클리어 중...")
        self._document_cache.clear()
        self._faiss_cache.clear()
        self._bm25_cache.clear()
        print("✅ 캐시가 클리어되었습니다.")

    def list_available_embeddings(self):
        print(f"📁 임베딩 디렉토리: {self.embeddings_dir}")
        if not os.path.exists(self.embeddings_dir):
            print("❌ 임베딩 디렉토리가 존재하지 않습니다.")
            return
        files = os.listdir(self.embeddings_dir)
        if not files:
            print("📂 임베딩 파일이 없습니다.")
            return
        print("📋 발견된 임베딩 파일:")
        for file in sorted(files):
            path = os.path.join(self.embeddings_dir, file)
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"   - {file} ({size:.2f} MB)")


if __name__ == "__main__":
    print("🚀 FAISSRAGRetriever 테스트 시작")
    rag = FAISSRAGRetriever()
    rag.list_available_embeddings()

    build_embeddings = input("\n💡 카테고리별 임베딩을 빌드하시겠습니까? (y/N): ").lower().strip()
    if build_embeddings in ["y", "yes"]:
        force_rebuild = input("🔄 기존 임베딩을 강제로 재빌드하시겠습니까? (y/N): ").lower().strip() in ["y", "yes"]
        rag.build_category_embeddings(force_rebuild=force_rebuild)

    latest_card = rag.get_latest_card_from_selected_cards()
    test_card = latest_card or "K-패스카드"
    if not latest_card:
        print(f"⚠️ selected_cards.json에서 카드를 가져올 수 없어 기본값 사용: {test_card}")

    q = "이 카드의 연회비는 얼마인가요?"
    try:
        print("\n🔍 기본 답변 테스트")
        print(rag.query(card_name=test_card, card_text="", question=q))

        print("\n🔍 쉬운 설명 테스트")
        print(rag.query(card_name=test_card, card_text="", question=q, explain_easy=True))
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
