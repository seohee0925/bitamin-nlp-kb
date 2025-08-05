# KB 카드 추천 RAG 시스템

KB국민은행 카드 상품을 추천하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

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
