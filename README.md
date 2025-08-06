# KB 카드 추천 RAG 시스템

KB국민은행 카드 상품을 추천하는 AI 시스템입니다.

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
2. **질문 입력**: "대중교통 혜택이 좋은 카드 추천해줘"
3. **카드 선택**: 추천된 카드 중 선택하여 상세 정보 확인

## 📝 예시 질문

- "대중교통 혜택이 좋은 카드 추천해줘"
- "연회비가 없는 카드 찾아줘"
- "쇼핑 할인 혜택이 있는 카드 알려줘"
- "외식 할인이 많은 카드 추천해줘"
