# Liiv Fit: 당신의 일상과 혜택이 만나는 순간, 쉽고 명확한 안내까지
<img width="947" height="518" alt="image" src="https://github.com/user-attachments/assets/c054e6ea-aeae-4840-ae18-aa3522d6152e" />

이 프로젝트는 **Dual RAG** 구조를 기반으로, 카드 추천과 상세 설명을 단계적으로 제공하는 자연어 기반 챗봇 시스템입니다. **Summary RAG**는 사용자가 입력한 요구사항과 가장 유사한 카드들을 FAISS 기반 검색으로 찾아 추천하고, 각 카드의 추천 이유와 비교 분석을 제공합니다.

이후 사용자가 특정 카드를 선택하면 **Original RAG**가 해당 카드의 PDF 약관 및 세부 혜택 정보를 분석하여 핵심 조건과 예외 조항을 명확하게 전달합니다. 또한 사용자 요구에 따라 복잡한 금융 용어나 긴 문장도 쉽게 풀어 설명함으로써, 고객이 카드 약관을 빠르고 정확하게 이해하고 합리적인 결정을 내릴 수 있도록 지원합니다.

## 🚀 빠른 시작

### 1. 저장소 다운로드
```bash
git clone https://github.com/seohee0925/bitamin-nlp-kb.git
cd bitamin-nlp-kb
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. API 키 설정
```bash
# macOS/Linux
export OPENAI_API_KEY="sk-your-api-key-here"

# Windows
set OPENAI_API_KEY=sk-your-api-key-here
```

### 4. 실행
```bash
cd RAG
python card_generator.py
```

## 💡 사용법

1. **카드 타입 선택**: 전체/신용카드/체크카드
2. **질문 입력**: 원하는 카드 조건에 대해서서 입력
3. **카드 선택**: 추천된 카드 중 선택하여 상세 정보 확인
4. **상세 질의응답**: 선택한 카드에 대해서 구체적으로 질문 및 응답
5. **쉽게 설명**: 답변에 대한 쉬운 설명 요청

## 📝 예시 질문
- "나는 40대 주부이고, 대형마트·온라인몰에서 장을 많이 봐. 마트·온라인 쇼핑 할인 카드가 필요해."
- "카페 결제와 편의점 이용이 많아. 커피·편의점 혜택 카드가 좋아."
- "연회비 부담은 낮았으면 하고, 포인트보다 즉시 할인을 선호해. 저연회비 할인형 카드를 찾고 있어."

## 🏗️ 시스템 아키텍처

이 프로젝트는 **Dual RAG** 구조를 기반으로, 카드 추천과 상세 설명을 단계적으로 제공하는 자연어 기반 챗봇 시스템입니다.


### Summary RAG
- 사용자 요구사항과 가장 유사한 카드들을 FAISS 기반 검색으로 찾아 추천
- 각 카드의 추천 이유와 비교 분석 제공
<img width="1009" height="385" alt="image" src="https://github.com/user-attachments/assets/33fbef04-fc9d-4a9c-b4f4-d4a39ce883e9" />


### Original RAG
- 선택된 카드의 PDF 약관 및 세부 혜택 정보를 분석
- 핵심 조건과 예외 조항을 명확하게 전달
- 복잡한 금융 용어를 쉽게 풀어서 설명
<img width="832" height="397" alt="image" src="https://github.com/user-attachments/assets/a9004e49-854e-45d3-8fd4-f190e30dacba" />


## 🗂️ 프로젝트 구조

```
📁 bitamin-nlp-kb
├── 📁 JSON
│   ├── 🗂️ Original JSON
│   │   ├── 🗂️ 신용json
│   │   └── 🗂️ 체크json
│   └── 🗂️ Summary JSON
│       ├── 🗂️ 신용json
│       └── 🗂️ 체크json
│
├── 📁 Embeddings
│   ├── 🗂️ sep_embeddings
│   ├── cards_summary_with_intro.json
│   └── embed_cards_separated.py
│
├── 📁 Summary RAG
│   ├── prompt.txt
│   ├── summary.py
│   ├── faiss_retriever.py
│   └── card_generator.py
│
├── 📁 Original RAG
│   ├── selected_cards.json
│   └── original_rag.py
│
├── 📁 UI
│   ├── app.py
│   └── example.html
│
└── 📄 requirements.txt
```

## 📁 주요 구성 요소

### JSON 데이터
- **Original JSON**: KB국민카드(신용/체크) 상품설명서·주요거래조건 PDF를 원문 구조 그대로 JSON화한 파일로, 섹션/항목(혜택, 조건, 연회비 등)을 보존한 근거 데이터입니다.
- **Summary JSON**: Original JSON(원문 약관 데이터)을 기반으로, LLM을 통해 카드명·브랜드·주요 혜택·적용 조건·대상 고객·연회비 등의 핵심 정보만 발췌한 요약본입니다. *(Dual RAG에서 추천 단계의 빠른 검색과 응답 생성을 위해 사용)*


### Embeddings
- `sep_embeddings`: 카드 추천 챗봇의 문서 검색(Retrieval) 기능을 위한 임베딩 데이터가 저장되어 있으며 카드별로 텍스트 데이터를 벡터화한 FAISS 인덱스로 구축한 결과물이 포함되어 있습니다
- `cards_summary_with_intro.json`: 카드 요약 정보와 소개 데이터 파일
- `embed_cards_separated.py`: 임베딩 생성 스크립트(신용/체크카드 분리)


### RAG 시스템
- **Summary RAG**:
  - `prompt.txt`: Summary RAG 구축을 위해 카드 상품 설명서·약관 JSON을 핵심 항목별로 자동 요약하는 프롬프트 txt 파일
  - `summary.py`: GPT-4o를 사용해 Summary RAG를 위한 카드 약관 JSON 요약본을 생성하고 저장하는 py 파일
  - `faiss_retriever.py`: 저장된 카드 임베딩을 불러온 뒤, 사용자의 질문을 벡터화하여 FAISS 코사인 유사도 기반으로 가장 관련성 높은 카드를 찾아주는 Summary RAG의 retriever 역할을 수행하는 대화형 검색 파일
  - `card_generator.py`: 사용자가 카드 관련 질문을 입력하면 Summary RAG의 retriever가 FAISS 기반 검색으로 상위 3개 카드 후보를 찾고, GPT로 추천·비교 분석을 생성하며, 선택된 카드는 Original RAG를 통해 상세 약관·혜택 정보를 검색·생성하는 콘솔형 카드 추천 메인 실행 파일

  
- **Original RAG**:
  - `selected_cards.json`: 사용자에게 맞춤형으로 추천된 카드들에 대한 하나의 예시
  - `original_rag.py`: 사용자의 질의를 받아 FAISS, BM25, RRF, Crossencoder Reranker를 통해 가장 관련성 높은 문서를 찾아내고, 이를 GPT-4o에 전달해 1차 응답을 생성한 뒤 GPT-4로 한 번 더 다듬어 최종적으로 명확하고 이해하기 쉬운 답변을 제공하는 파일


### UI
- `app.py`: FastAPI 기반의 백엔드 엔트리포인트로, 루트(/)에서 example.html을 렌더링하고 /recommend(POST)에서 사용자 입력을 받아 RAG Generator + Rewrite를 실행해 응답과 후속 질문을 관리합니다. 동시에 정적 파일(/static)을 서빙하고 .env 환경변수도 로드하며, 로컬 개발은 uvicorn app:app --reload 명령으로 바로 실행할 수 있도록 구성되어 있습니다
- `example.html`: Jinja2 템플릿의 챗봇 UI로, 대화 말풍선을 자연스럽게 렌더링하고 사용자 입력 폼과 "쉽게 설명", "이 답변에 대해 질문하기" 버튼을 제공하며, 로딩(대기) 상태와 카드 추천 섹션까지 한 화면에서 보여줍니다. 서버에서 전달된 chat_history를 그대로 반영해 이전 대화 맥락을 이어주도록 설계되어 있습니다

## 🔧 기술 스택
- **Data Processing**: JSON, PDF Processing
- **Vector Search**: FAISS, Cosine Similarity
- **AI/ML**: OpenAI GPT, FAISS, BM25, Crossencoder Reranker
- **Frontend**: Python, FastAPI

## 👥 역할 분담

<table>
  <thead>
    <tr>
      <th>팀원</th>
      <th>역할</th>
      <th>담당 업무</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>윤서희</td>
      <td>팀장</td>
      <td>데이터 수집 및 전처리, Original RAG 개발(Retriever)</td>
    </tr>
    <tr>
      <td>서준형</td>
      <td>팀원</td>
      <td>데이터 수집 및 전처리, Summary RAG 개발</td>
    </tr>
    <tr>
      <td>안유민</td>
      <td>팀원</td>
      <td>데이터 수집 및 전처리, Original RAG 개발(Generator), 챗봇 구현</td>
    </tr>
  </tbody>
</table>

## 🎯 주요 기능

- **맞춤형 카드 추천**: 사용자 요구사항에 기반한 정확한 카드 추천
- **비교 분석**: 추천된 카드들의 상세 비교 및 분석
- **약관 해석**: 복잡한 금융 약관을 쉽게 풀어서 설명
- **대화형 인터페이스**: 자연어 기반의 직관적인 사용자 경험
- **실시간 검색**: FAISS 기반의 빠른 벡터 검색

---

**KB 카드 추천 시스템**으로 최적의 카드를 찾아보세요! 💳 <br>

🎥 실제 작동하는 KB 카드 추천 시스템을 영상으로 확인하세요!
사용자 질문부터 카드 추천, 상세 정보 제공까지의 전체 과정을 데모로 보실 수 있습니다.
https://www.youtube.com/watch?v=Piwf24nMLa4&feature=youtu.be
