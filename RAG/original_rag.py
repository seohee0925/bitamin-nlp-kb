import os
import re
import json
import hashlib
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from collections import defaultdict
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import numpy as np


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
        """Original RAG 시스템 초기화"""
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        load_dotenv()
        
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.dirname(self.current_file_dir)  # 프로젝트 루트 폴더
        self.data_dirs = [
            os.path.join(self.base_path, "JSON", "신용json"),
            os.path.join(self.base_path, "JSON", "체크json")
        ]

        self.selected_cards_path = os.path.join(self.current_file_dir, "selected_cards.json")
        
        # 임베딩 저장 디렉토리 생성
        self.embeddings_dir = os.path.join(self.current_file_dir, "original_embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        print(f"📁 임베딩 저장 디렉토리: {self.embeddings_dir}")
        
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", 
            model_kwargs={"device": "cpu"}
        )
        
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.graph = self._build_langgraph()
        
        # 카드별 캐시된 문서와 인덱스 저장
        self._document_cache = {}
        self._faiss_cache = {}
        self._bm25_cache = {}
        
        print("🎉 FAISSRAGRetriever 초기화 완료!")

    def get_latest_card_from_selected_cards(self) -> Optional[str]:
        """selected_cards.json에서 마지막 카드 이름 가져오기"""
        
        try:
            if not os.path.exists(self.selected_cards_path):
                print(f"❌ selected_cards.json 파일을 찾을 수 없습니다: {self.selected_cards_path}")
                return None
            
            with open(self.selected_cards_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                print("❌ selected_cards.json이 비어있습니다")
                return None
            
            # 마지막 항목 가져오기
            last_card = data[-1]
            card_name = last_card.get("card_name")
            
            print(f"🎯 마지막 카드 정보:")
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
        """Field 단위로 문서 로딩"""
        print(f"📋 문서 로딩 시작... (총 {len(json_paths)}개 파일)")
        documents = []
        
        for json_path in tqdm(json_paths, desc="📋 문서 로딩 중"):
            try:
                print(f"📄 파일 처리 중: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                card_name = data.get("card_name", "UnknownCard")
                print(f"   - 카드명: {card_name}")
                
                sections_count = 0
                for section in data.get("sections", []):
                    heading = section.get("heading", "")
                    subheading = section.get("subheading", "")
                    
                    # 다양한 필드에서 내용 추출
                    for key in ['benefit', 'benefits', 'fee', 'agreement', 'condition', 'conditions', 'etc', 'overseas_usage']:
                        val = section.get(key)
                        if not val:
                            continue
                            
                        contents = val if isinstance(val, list) else [val]
                        merged_text = "\n".join([c.strip() for c in contents if len(c.strip()) > 0])
                        
                        if len(merged_text.strip()) < 10:
                            continue
                            
                        full_text = f"[{card_name}]\n{heading} - {subheading}\n<{key}>\n{merged_text}"
                        documents.append(Document(
                            page_content=full_text,
                            metadata={
                                "card_name": card_name, 
                                "field": key, 
                                "heading": heading, 
                                "subheading": subheading
                            }
                        ))
                        sections_count += 1      
                        
            except Exception as e:
                print(f"❗ Error reading {json_path}: {e}")

        return documents

    def _get_card_category(self, json_path: str) -> str:
        """카드 JSON 파일 경로에서 카테고리(신용/체크) 판별"""
        if "신용json" in json_path:
            return "credit"
        elif "체크json" in json_path:
            return "check"
        else:
            return "unknown"

    def _load_category_documents(self, category: str) -> list[Document]:
        """카테고리별로 모든 문서 로딩 (신용카드 or 체크카드)"""
        print(f"📋 {category.upper()} 카드 카테고리 문서 로딩 중...")
        
        target_dir = None
        if category == "credit":
            target_dir = os.path.join(self.base_path, "JSON", "신용json")
        elif category == "check":
            target_dir = os.path.join(self.base_path, "JSON", "체크json")
        else:
            raise ValueError(f"지원하지 않는 카테고리: {category}")
        
        if not os.path.exists(target_dir):
            print(f"❌ 카테고리 디렉토리를 찾을 수 없습니다: {target_dir}")
            return []
        
        # 모든 JSON 파일 경로 수집
        json_files = []
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        print(f"📊 {category.upper()} 카테고리에서 {len(json_files)}개 JSON 파일 발견")
        
        if not json_files:
            print(f"⚠️ {category.upper()} 카테고리에 JSON 파일이 없습니다")
            return []
        
        return self.load_documents_field_level(json_files)

    def _save_category_embeddings(self, category: str, documents: list[Document], faiss_index: FAISS) -> bool:
        """카테고리별 임베딩 데이터 저장"""
        print(f"💾 {category.upper()} 카드 임베딩 저장 중...")
        
        try:
            # 파일 경로 정의
            base_filename = f"{category}_card"
            faiss_path = os.path.join(self.embeddings_dir, f"{base_filename}_embeddings.faiss")
            pkl_path = os.path.join(self.embeddings_dir, f"{base_filename}_embedding_data.pkl")
            metadata_path = os.path.join(self.embeddings_dir, f"{base_filename}_metadata.json")
            texts_path = os.path.join(self.embeddings_dir, f"{base_filename}_texts.txt")
            
            print(f"   📁 저장 경로:")
            print(f"      - FAISS 인덱스: {faiss_path}")
            print(f"      - 임베딩 데이터: {pkl_path}")
            print(f"      - 메타데이터: {metadata_path}")
            print(f"      - 텍스트: {texts_path}")
            
            # 1. FAISS 인덱스만 저장 (langchain의 복잡한 구조 대신)
            if hasattr(faiss_index, 'index'):
                import faiss as faiss_lib
                faiss_lib.write_index(faiss_index.index, faiss_path)
            
            # 2. 문서 데이터를 pickle로 저장
            embedding_data = {
                'documents': documents,
                'texts': [doc.page_content for doc in documents],
                'metadatas': [doc.metadata for doc in documents],
                'embeddings': faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal) if hasattr(faiss_index.index, 'reconstruct_n') else None
            }
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(embedding_data, f)
            print("   ✅ 임베딩 데이터 저장 완료")
            
            # 3. 메타데이터를 JSON으로 저장
            import datetime
            metadata_summary = {
                'category': category,
                'total_documents': len(documents),
                'cards': list(set([doc.metadata.get('card_name', 'Unknown') for doc in documents])),
                'created_at': str(datetime.datetime.now().isoformat()),
                'embedding_model': 'BAAI/bge-m3'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
            print("   ✅ 메타데이터 저장 완료")
            
            # 4. 텍스트 데이터를 txt로 저장 (검수용)
            with open(texts_path, 'w', encoding='utf-8') as f:
                f.write(f"=== {category.upper()} 카드 문서 텍스트 ===\n")
                f.write(f"총 문서 수: {len(documents)}\n")
                f.write(f"포함된 카드: {', '.join(metadata_summary['cards'])}\n")
                f.write("="*50 + "\n\n")
                
                for i, doc in enumerate(documents):
                    f.write(f"문서 {i+1}:\n")
                    f.write(f"카드명: {doc.metadata.get('card_name', 'Unknown')}\n")
                    f.write(f"필드: {doc.metadata.get('field', 'Unknown')}\n")
                    f.write(f"제목: {doc.metadata.get('heading', '')} - {doc.metadata.get('subheading', '')}\n")
                    f.write("-"*30 + "\n")
                    f.write(doc.page_content[:500] + "...\n" if len(doc.page_content) > 500 else doc.page_content + "\n")
                    f.write("\n" + "="*50 + "\n\n")
            
            print(f"✅ {category.upper()} 카드 임베딩 저장 성공!")
            return True
            
        except Exception as e:
            print(f"❌ {category.upper()} 카드 임베딩 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_category_embeddings(self, category: str) -> tuple[list[Document], FAISS, BM25Retriever]:
        """카테고리별 임베딩 로드"""
        
        base_filename = f"{category}_card"
        faiss_path = os.path.join(self.embeddings_dir, f"{base_filename}_embeddings.faiss")
        pkl_path = os.path.join(self.embeddings_dir, f"{base_filename}_embedding_data.pkl")
        
        # 파일 존재 확인
        if not os.path.exists(pkl_path):
            print(f"⚠️ {category.upper()} 카드 임베딩 파일이 없습니다: {pkl_path}")
            return None, None, None
        
        try:
            # 문서 데이터 로드
            with open(pkl_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            documents = embedding_data['documents']
            
            # FAISS 인덱스 재구성
            if os.path.exists(faiss_path):
                # FAISS 파일이 있다면 로드
                import faiss as faiss_lib
                index = faiss_lib.read_index(faiss_path)
                
                # Langchain FAISS 객체 재구성
                faiss_index = FAISS(
                    embedding_function=self.embedding_model,
                    index=index,
                    docstore={i: doc for i, doc in enumerate(documents)},
                    index_to_docstore_id={i: i for i in range(len(documents))}
                )
                print(f"   🔍 FAISS 인덱스 로드 완료")
            else:
                # FAISS 파일이 없다면 다시 생성
                print(f"   ⚠️ FAISS 파일이 없어서 다시 생성합니다...")
                faiss_index = FAISS.from_documents(documents, self.embedding_model)
            
            # BM25 인덱스 생성
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
        """신용카드/체크카드 카테고리별 임베딩 빌드"""
        
        for category in ["credit", "check"]:
            print(f"\n{'='*50}")
            print(f"{'='*50}")
            
            base_filename = f"{category}_card"
            pkl_path = os.path.join(self.embeddings_dir, f"{base_filename}_embedding_data.pkl")
            
            # 이미 존재하는지 확인
            if os.path.exists(pkl_path) and not force_rebuild:
                print(f"✅ {category.upper()} 카드 임베딩이 이미 존재합니다: {pkl_path}")
                print("   force_rebuild=True로 설정하면 재빌드됩니다.")
                continue
            
            # 카테고리별 문서 로딩
            documents = self._load_category_documents(category)
            
            if not documents:
                print(f"⚠️ {category.upper()} 카테고리에 문서가 없습니다. 스킵합니다.")
                continue
            
            print(f"📊 {category.upper()} 카테고리 통계:")
            cards = list(set([doc.metadata.get('card_name', 'Unknown') for doc in documents]))
            print(f"   - 총 문서 수: {len(documents)}")
            print(f"   - 카드 수: {len(cards)}")
            print(f"   - 포함된 카드: {', '.join(cards[:5])}{'...' if len(cards) > 5 else ''}")
            
            # FAISS 인덱스 생성
            print(f"🔍 {category.upper()} 카드 FAISS 인덱스 생성 중...")
            faiss_index = FAISS.from_documents(documents, self.embedding_model)
            print(f"✅ {category.upper()} 카드 FAISS 인덱스 생성 완료")
            
            # 임베딩 저장
            success = self._save_category_embeddings(category, documents, faiss_index)
            
            if success:
                print(f"🎉 {category.upper()} 카드 임베딩 빌드 완료!")
            else:
                print(f"❌ {category.upper()} 카드 임베딩 빌드 실패!")
        
        print(f"📁 저장 위치: {self.embeddings_dir}")

    def reciprocal_rank_fusion(self, faiss_results: list, bm25_results: list, k: int = 60) -> list[str]:
        """RRF Score 계산으로 결과 융합"""
        scores = defaultdict(float)
        
        def update_scores(results, weight):
            for rank, item in enumerate(results):
                key = item.page_content.strip()
                scores[key] += weight / (k + rank + 1)
        
        update_scores(faiss_results, weight=0.6)
        update_scores(bm25_results, weight=0.4)
        
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result_chunks = [chunk for chunk, _ in sorted_chunks[:k]]
        return result_chunks

    def build_generator_prompt(self, card_name: str, user_question: str, context_chunks: list[str]) -> str:
        """답변 생성용 프롬프트 구성"""
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

    # LangGraph 노드 정의
    def _build_prompt_node(self, state: GeneratorState) -> GeneratorState:
        print("🔧 프롬프트 노드 실행 중...")
        state["prompt"] = self.build_generator_prompt(
            card_name=state["card_name"],
            user_question=state["user_question"],
            context_chunks=state["context_chunks"]
        )
        return state

    def _generate_answer_node(self, state: GeneratorState) -> GeneratorState:
        messages = [
            SystemMessage(content="당신은 카드별 정보 기반 응답을 정확하게 생성하는 전문가입니다."),
            HumanMessage(content=state["prompt"])
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
4. 너무 단순화하거나, 말을 지어내거나, 법적 표현을 삭제하지 말고 **문맥 그대로 쉽게 풀어** 쓰세요.
5. 전체적으로 **고객 상담원이 친절하게 설명해주는 말투**로 바꾸세요.

[원문]
{state['answer']}

[쉬운 설명]
"""
        messages = [
            SystemMessage(content="당신은 금융 정보를 쉽게 설명해주는 AI입니다."),
            HumanMessage(content=prompt)
        ]
        state["simplified_answer"] = self.llm(messages).content
        print(f"✅ 쉬운 설명 생성 완료 (길이: {len(state['simplified_answer'])}자)")
        return state

    def _build_langgraph(self) -> StateGraph:
        """LangGraph 구성"""
        print("🔧 LangGraph 구성 중...")
        builder = StateGraph(GeneratorState)
        builder.add_node("BuildPrompt", self._build_prompt_node)
        builder.add_node("GenerateAnswer", self._generate_answer_node)
        builder.add_node("RewriteAnswer", self._rewrite_answer_node)
        builder.set_entry_point("BuildPrompt")
        builder.add_edge("BuildPrompt", "GenerateAnswer")
        builder.add_edge("GenerateAnswer", "RewriteAnswer")
        builder.add_edge("RewriteAnswer", END)
        print("✅ LangGraph 구성 완료")
        return builder.compile()

    def _find_card_json_path(self, card_name: str) -> Optional[str]:
        """카드 이름으로 JSON 파일 경로 찾기 (카테고리 폴더 포함 재귀 검색)"""
        print(f"🔍 '{card_name}' 카드의 JSON 파일 검색 중...")
        
        for data_dir in self.data_dirs:
            print(f"📂 메인 디렉토리 검색: {data_dir}")
            if not os.path.exists(data_dir):
                print(f"⚠️ 디렉토리를 찾을 수 없습니다: {data_dir}")
                continue
            
            # 디렉토리 내용 확인
            try:
                items = os.listdir(data_dir)
                print(f"   📋 디렉토리 내용: {items}")
            except Exception as e:
                print(f"   ❗ 디렉토리 읽기 오류: {e}")
                continue
            
            # 재귀적으로 모든 하위 디렉토리와 파일 검색
            total_files_checked = 0
            categories_found = []
            
            for root, dirs, files in os.walk(data_dir):
                # 현재 검색 중인 카테고리 폴더 표시
                relative_path = os.path.relpath(root, data_dir)
                if relative_path != ".":
                    categories_found.append(relative_path)
                
                json_files = [f for f in files if f.endswith('.json')]
                total_files_checked += len(json_files)
                
                if json_files:
                    print(f"      - JSON 파일 {len(json_files)}개 발견: {json_files[:3]}{'...' if len(json_files) > 3 else ''}")
                
                for filename in json_files:
                    try:
                        filepath = os.path.join(root, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            file_card_name = data.get("card_name")
                            print(f"            - 파일 내 카드명: '{file_card_name}'")
                            
                            if file_card_name == card_name:
                                print(f"✅ 매칭되는 파일 발견!")
                                print(f"   - 파일 경로: {filepath}")
                                print(f"   - 파일의 카드명: {file_card_name}")
                                print(f"   - 카테고리: {relative_path}")
                                return filepath
                                
                    except Exception as e:
                        print(f"❗ JSON 파일 읽기 오류 {filename}: {e}")
                        continue
            
            print(f"📊 '{os.path.basename(data_dir)}' 검색 요약:")
            print(f"   - 총 검사한 JSON 파일: {total_files_checked}개")
            print(f"   - 발견된 카테고리: {categories_found}")

        self._show_available_cards_sample()
        
        return None
    
    def _show_available_cards_sample(self, max_samples: int = 5):
        """실제 존재하는 카드들의 샘플 목록 표시 (디버깅용)"""
        found_cards = []
        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                continue
                
            for root, dirs, files in os.walk(data_dir):
                json_files = [f for f in files if f.endswith('.json')]
                
                for filename in json_files[:max_samples]:  # 샘플만 확인
                    try:
                        filepath = os.path.join(root, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            card_name = data.get("card_name")
                            if card_name:
                                category = os.path.relpath(root, data_dir)
                                found_cards.append((card_name, category))
                                print(f"   - '{card_name}' (카테고리: {category})")
                    except:
                        continue
                        
                if len(found_cards) >= max_samples:
                    break
            
            if len(found_cards) >= max_samples:
                break
        
        if not found_cards:
            print("   ❌ 사용 가능한 카드를 찾을 수 없습니다.")
        else:
            print(f"   💡 총 {len(found_cards)}개 샘플 표시 (실제로는 더 많을 수 있음)")

    def _get_card_category_from_path(self, json_path: str) -> str:
        """JSON 파일 경로에서 카드 카테고리 추출"""
        if "신용json" in json_path or "credit" in json_path.lower():
            return "credit"
        elif "체크json" in json_path or "check" in json_path.lower():
            return "check"
        else:
            return "unknown"

    def _prepare_card_data(self, card_name: str) -> tuple[list[Document], FAISS, BM25Retriever]:
        """카드별 문서 및 인덱스 준비 (카테고리별 캐싱 + 개별 카드 필터링)"""
        print(f"🔧 '{card_name}' 카드 데이터 준비 중...")
        
        # 개별 카드 캐시 확인
        if card_name in self._document_cache:
            print("💾 개별 카드 캐시 사용 중...")
            return (
                self._document_cache[card_name],
                self._faiss_cache[card_name],
                self._bm25_cache[card_name]
            )

        # 카드 JSON 파일 찾기
        json_path = self._find_card_json_path(card_name)
        if not json_path:
            raise ValueError(f"'{card_name}' 카드의 데이터를 찾을 수 없습니다.")

        # 카드의 카테고리 판별
        category = self._get_card_category_from_path(json_path)

        # 카테고리별 임베딩이 있는지 확인하고 로드
        category_documents, category_faiss, category_bm25 = self._load_category_embeddings(category)
        
        if category_documents is None:
            print(f"⚠️ {category.upper()} 카테고리 임베딩이 없습니다. 개별 카드로 처리합니다...")
            return self._prepare_individual_card_data(card_name, json_path)
        
        # 해당 카드의 문서만 추출
        card_documents = [doc for doc in category_documents if doc.metadata.get('card_name') == card_name]
        
        if not card_documents:
            print(f"❌ {category.upper()} 카테고리에서 '{card_name}' 카드를 찾을 수 없습니다.")
            raise ValueError(f"카테고리 임베딩에 '{card_name}' 카드가 없습니다.")

        card_faiss = FAISS.from_documents(card_documents, self.embedding_model)
        card_bm25 = BM25Retriever.from_documents(card_documents)
        card_bm25.k = 60

        # 개별 카드 캐시 저장
        self._document_cache[card_name] = card_documents
        self._faiss_cache[card_name] = card_faiss
        self._bm25_cache[card_name] = card_bm25

        return card_documents, card_faiss, card_bm25

    def _prepare_individual_card_data(self, card_name: str, json_path: str) -> tuple[list[Document], FAISS, BM25Retriever]:
        """개별 카드 데이터 준비 (기존 방식)"""
        documents = self.load_documents_field_level([json_path])
        
        if not documents:
            raise ValueError(f"'{card_name}' 카드의 문서가 비어 있습니다.")

        faiss_index = FAISS.from_documents(documents, self.embedding_model)
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 60
        return documents, faiss_index, bm25

    def query(self, card_name: str, card_text: str, question: str, explain_easy: bool = False, top_k: int = 20) -> str:
        """
        카드에 대한 질의응답 수행 (card_generator.py 호환용 인터페이스)
        
        Args:
            card_name: 카드 이름 (selected_cards.json에서 가져온 값)
            card_text: 카드 텍스트 (사용되지 않음, 호환성을 위해 유지)
            question: 사용자 질문
            explain_easy: 쉽게 설명할지 여부
            top_k: 상위 k개 문서 사용
            
        Returns:
            답변 텍스트
        """
        print(f"\n{'='*60}")
        print(f"🚀 질의응답 시작")
        print(f"   - 카드명: {card_name}")
        print(f"   - 질문: {question}")
        print(f"   - 쉬운 설명: {explain_easy}")
        print(f"   - Top-K: {top_k}")
        print(f"{'='*60}")
        
        try:
            print(f"🔍 '{card_name}' 카드 정보 검색 중...")
            
            # 카드 데이터 준비
            documents, faiss_index, bm25 = self._prepare_card_data(card_name)

            print("📊 문서 검색 및 랭킹 중...")
            # FAISS 검색
            print("   - FAISS 유사도 검색 실행...")
            faiss_results = faiss_index.similarity_search(question, k=60)
            print(f"   - FAISS 결과: {len(faiss_results)}개")
            
            # BM25 검색
            print("   - BM25 검색 실행...")
            bm25_results = bm25.get_relevant_documents(question)
            print(f"   - BM25 결과: {len(bm25_results)}개")
            
            # RRF로 결과 융합
            rrf_candidates = self.reciprocal_rank_fusion(faiss_results, bm25_results)
            
            # Cross-Encoder로 재랭킹
            print(f"🔄 Cross-Encoder 재랭킹 실행... (총 {len(rrf_candidates)}개 후보)")
            inputs = [(question, chunk) for chunk in rrf_candidates]
            scores = self.reranker.predict(inputs)
            reranked_chunks = [x for _, x in sorted(zip(scores, rrf_candidates), reverse=True)]
            top_chunks = reranked_chunks[:top_k]
            print(f"✅ 재랭킹 완료 - 최종 {len(top_chunks)}개 청크 선택")

            # LangGraph로 답변 생성
            print("💭 LangGraph 답변 생성 시작...")
            initial_state = {
                "card_name": card_name,
                "user_question": question,
                "context_chunks": top_chunks,
                "prompt": "",
                "answer": "",
                "simplified_answer": "",
                "explain_easy": explain_easy
            }

            result = self.graph.invoke(initial_state)
            
            # 쉬운 설명이 요청되었고 생성되었다면 그것을 반환
            final_answer = ""
            if explain_easy and result["simplified_answer"]:
                final_answer = result["simplified_answer"]
                print("📝 쉬운 설명 버전 반환")
            else:
                final_answer = result["answer"]
                print("📝 기본 답변 버전 반환")
            
            print(f"🎉 질의응답 완료! (최종 답변 길이: {len(final_answer)}자)")
            return final_answer

        except Exception as e:
            error_msg = f"❌ '{card_name}' 카드 질의응답 중 오류가 발생했습니다: {str(e)}"
            print(error_msg)
            return error_msg

    def clear_cache(self):
        """캐시된 데이터 클리어"""
        print("🗑️ 캐시 클리어 중...")
        self._document_cache.clear()
        self._faiss_cache.clear()
        self._bm25_cache.clear()
        print("✅ 캐시가 클리어되었습니다.")

    def list_available_embeddings(self):
        """사용 가능한 임베딩 파일 목록 표시"""
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
            filepath = os.path.join(self.embeddings_dir, file)
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"   - {file} ({size:.2f} MB)")


if __name__ == "__main__":
    print("🚀 FAISSRAGRetriever 테스트 시작")
    
    # 테스트 코드
    rag = FAISSRAGRetriever()
    
    # 사용 가능한 임베딩 확인
    rag.list_available_embeddings()
    
    # 카테고리별 임베딩 빌드 (처음 실행시에만)
    build_embeddings = input("\n💡 카테고리별 임베딩을 빌드하시겠습니까? (y/N): ").lower().strip()
    if build_embeddings in ['y', 'yes']:
        force_rebuild = input("🔄 기존 임베딩을 강제로 재빌드하시겠습니까? (y/N): ").lower().strip() in ['y', 'yes']
        rag.build_category_embeddings(force_rebuild=force_rebuild)
    
    # selected_cards.json에서 마지막 카드 가져오기
    latest_card = rag.get_latest_card_from_selected_cards()
    
    if latest_card:
        test_card = latest_card
        print(f"🎯 테스트할 카드: {test_card}")
    else:
        test_card = "K-패스카드"  # fallback
        print(f"⚠️ selected_cards.json에서 카드를 가져올 수 없어 기본값 사용: {test_card}")
    
    test_question = "이 카드의 연회비는 얼마인가요?"
    
    try:
        print(f"\n🔍 기본 답변 테스트")
        answer = rag.query(card_name=test_card, card_text="", question=test_question)
        print(f"\n질문: {test_question}")
        print(f"답변: {answer}")
        
        print(f"\n🔍 쉬운 설명 테스트")
        # 쉬운 설명 테스트
        easy_answer = rag.query(card_name=test_card, card_text="", question=test_question, explain_easy=True)
        print(f"쉬운 설명: {easy_answer}")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()