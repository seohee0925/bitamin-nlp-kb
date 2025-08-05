# KB 카드 추천 RAG 시스템

KB국민은행 카드 상품을 추천하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 📁 프로젝트 구조

KB/
├── RAG/                          # RAG 시스템 핵심 모듈
│   ├── card_generator.py         # 메인 카드 추천 시스템
│   ├── faiss_retriever.py        # FAISS 검색 엔진
│   ├── rag_system.py             # RAG 시스템 통합 모듈
│   └── selected_cards.json       # 선택된 카드 데이터
├── CSV/                          # 원본 CSV 데이터
│   ├── KB카드_전체데이터.csv     # KB 카드 전체 데이터
│   ├── 신용crawl/               # 신용카드 크롤링 데이터
│   ├── 체크crawl/               # 체크카드 크롤링 데이터
│   └── merge_cards.py           # 카드 데이터 병합 스크립트
├── JSON/                         # 구조화된 JSON 데이터
│   ├── 신용json/                # 신용카드 JSON 데이터
│   ├── 체크json/                # 체크카드 JSON 데이터
│   └── add_intro_to_json.py     # JSON 데이터 전처리
├── embeddings/                   # 임베딩 데이터
│   ├── sep_embeddings/          # 분리된 임베딩 파일
│   │   ├── 신용카드_card_embeddings.faiss
│   │   ├── 체크카드_card_embeddings.faiss
│   │   └── card_metadata.json
│   └── embed_cards_separated.py # 임베딩 생성 스크립트
├── summary/                      # 카드 요약 데이터
│   ├── 신용_summary/            # 신용카드 요약
│   ├── 체크_summary/            # 체크카드 요약
│   └── sum.py                   # 요약 생성 스크립트
├── requirements.txt              # 필요한 패키지 목록
└── README.md    

## 🚀 빠른 시작

### 1. 환경변수 설정

```bash
# macOS/Linux
export OPENAI_API_KEY="your_api_key_here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your_api_key_here
```

### 2. 시스템 실행

```bash
# 카드 추천 시스템 실행
python card_generator.py
```

## 📋 시스템 구성

- **`faiss_retriever.py`**: FAISS를 사용한 카드 검색 엔진
- **`card_generator.py`**: GPT-4o를 사용한 카드 추천 생성기

## 🔧 설치 및 설정

### 필수 패키지

```bash
pip install openai faiss-cpu numpy
```

### API 키 설정 (필수)

**환경변수로 설정해야 합니다:**

1. **macOS/Linux**
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

2. **Windows PowerShell**
   ```powershell
   $env:OPENAI_API_KEY="sk-your-api-key-here"
   ```

3. **Windows Command Prompt**
   ```cmd
   set OPENAI_API_KEY=sk-your-api-key-here
   ```

4. **영구 설정 (macOS/Linux)**
   ```bash
   # ~/.bashrc 또는 ~/.zshrc에 추가
   echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

## 💡 사용법

### 1. 카드 타입 선택
- **전체 (all)**: 모든 카드 검색
- **신용카드 (credit)**: 신용카드만 검색
- **체크카드 (check)**: 체크카드만 검색

### 2. 질문 입력
```
💬 질문을 입력하세요: 대중교통 혜택이 좋은 카드 추천해줘
```

### 3. 카드 선택
추천된 3개 카드 중 하나를 선택하여 추가 정보 확인:
- 다른 카드와 비교하기
- 자세한 혜택 설명
- 신청 방법 안내

## 📝 예시 질문

- "대중교통 혜택이 좋은 카드 추천해줘"
- "연회비가 없는 카드 찾아줘"
- "쇼핑 할인 혜택이 있는 카드 알려줘"
- "외식 할인이 많은 카드 추천해줘"
- "주유 할인 혜택이 있는 카드 찾아줘"
