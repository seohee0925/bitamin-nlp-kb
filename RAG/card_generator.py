import json
from openai import OpenAI
import os
from datetime import datetime
import re # 정규 표현식 모듈 추가

class CardGenerator:
    def __init__(self):
        """카드 답변 생성기 초기화"""
        self.client = None
        self.selected_cards = []  # 선택된 카드들을 저장할 리스트
        self.init_openai_client()
    
    def init_openai_client(self):
        """OpenAI 클라이언트 초기화"""
        # 환경변수에서 API 키 확인
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 환경변수를 설정해주세요.")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, question, search_results, card_type="all"):
        """검색 결과를 바탕으로 답변 생성"""
        
        # 검색 결과를 텍스트로 정리
        context = self.format_search_results(search_results)
        
        # 카드 타입에 따른 프롬프트 조정
        card_type_display = "전체" if card_type == "all" else "신용카드" if card_type == "credit" else "체크카드"
        
        # 프롬프트 생성
        prompt = f"""
당신은 KB 카드 상품 전문 상담사입니다. 
사용자의 질문에 대해 검색된 카드 정보를 바탕으로 친절하고 정확한 답변을 제공해주세요.

선택된 카드 타입: {card_type_display}
사용자 질문: {question}

검색된 카드 정보:
{context}

답변 작성 규칙:
1. 상위 3개 카드를 추천하고 각각의 추천 이유를 명확히 설명
2. 각 카드의 주요 혜택과 특징을 구체적으로 설명
3. 사용자의 질문과 카드의 연관성을 강조
4. 친근하고 이해하기 쉬운 언어 사용
5. 한국어로 답변

답변 형식:
🎯 추천 카드 1: [카드명]
📋 추천 이유: [구체적인 이유]
💡 주요 혜택: [핵심 혜택들]

🎯 추천 카드 2: [카드명]
📋 추천 이유: [구체적인 이유]
💡 주요 혜택: [핵심 혜택들]

🎯 추천 카드 3: [카드명]
📋 추천 이유: [구체적인 이유]
💡 주요 혜택: [핵심 혜택들]

답변:
"""
        
        # GPT-4o로 답변 생성
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 KB 카드 상품 전문 상담사입니다. 친절하고 정확한 답변을 제공해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
    
    def format_search_results(self, search_results):
        """검색 결과를 읽기 쉬운 형태로 포맷팅"""
        formatted_results = []
        
        for i, result in enumerate(search_results, 1):
            card_info = f"""
{i}위 카드: {result['card_name']}
- 유형: {result['card_type']}
- 키워드: {result['keyword']}
- 상세정보: {result['card_text']}
"""
            formatted_results.append(card_info)
        
        return "\n".join(formatted_results)
    
    def generate_comparison(self, search_results):
        """여러 카드를 비교하는 답변 생성"""
        
        if len(search_results) < 2:
            return "비교할 카드가 충분하지 않습니다."
        
        # 카드 정보를 구조화된 형태로 정리
        cards_info = []
        for result in search_results:
            card_info = {
                'name': result['card_name'],
                'type': result['card_type'],
                'keyword': result['keyword'],
                'similarity': float(result['similarity_score']),  # float32를 float로 변환
                'details': result['card_text'],
                'benefits': self.extract_benefits_from_text(result['card_text']),
                'annual_fee': self.extract_annual_fee(result['card_text'])
            }
            cards_info.append(card_info)
        
        # 비교 프롬프트 생성
        comparison_prompt = f"""
다음 {len(cards_info)}개 카드를 체계적으로 비교 분석하여 사용자에게 추천해주세요:

{json.dumps(cards_info, ensure_ascii=False, indent=2)}

분석 기준:
1. 혜택의 다양성과 실용성
2. 연회비 대비 혜택 가치
3. 사용자 상황별 적합성
4. 조건의 합리성

답변 형식:
📊 카드 비교 분석 결과
=====================================

🏆 1순위: [카드명]
✅ 주요 혜택: [핵심 혜택들]
💰 연회비: [연회비]
💡 추천 대상: [어떤 사용자에게 적합한지]

🥈 2순위: [카드명]
✅ 주요 혜택: [핵심 혜택들]
💰 연회비: [연회비]
💡 추천 대상: [어떤 사용자에게 적합한지]

🥉 3순위: [카드명]
✅ 주요 혜택: [핵심 혜택들]
💰 연회비: [연회비]
💡 추천 대상: [어떤 사용자에게 적합한지]

📈 종합 평가
=====================================
• 가장 실용적인 카드: [카드명] - [이유]
• 가장 경제적인 카드: [카드명] - [이유]
• 가장 범용적인 카드: [카드명] - [이유]

💡 선택 가이드
=====================================
[사용자 상황별 추천 카드와 이유]

답변:
"""
        
        # GPT-4o로 비교 분석 생성
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 카드 상품 비교 분석 전문가입니다. 객관적이고 체계적인 비교 분석을 제공해주세요."},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
    
    def extract_benefits_from_text(self, card_text):
        """카드 텍스트에서 혜택 정보 추출"""
        benefits = []
        
        # 할인율 패턴 매칭
        discount_patterns = [
            r'(\d+(?:\.\d+)?)%\s*할인',
            r'할인\s*(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%\s*캐시백',
            r'캐시백\s*(\d+(?:\.\d+)?)%'
        ]
        
        for pattern in discount_patterns:
            matches = re.findall(pattern, card_text)
            for match in matches:
                benefits.append(f"{match}% 할인/캐시백")
        
        # 특정 혜택 패턴 매칭
        specific_benefits = [
            r'(\d+(?:,\d+)?)원\s*할인',
            r'할인\s*(\d+(?:,\d+)?)원',
            r'(\d+(?:,\d+)?)원\s*캐시백',
            r'캐시백\s*(\d+(?:,\d+)?)원'
        ]
        
        for pattern in specific_benefits:
            matches = re.findall(pattern, card_text)
            for match in matches:
                benefits.append(f"{match}원 할인/캐시백")
        
        return benefits if benefits else ["혜택 정보 없음"]
    
    def extract_annual_fee(self, card_text):
        """카드 텍스트에서 연회비 추출"""
        annual_fee_match = re.search(r'연회비[:\s]*([0-9,]+)원?', card_text)
        if annual_fee_match:
            return annual_fee_match.group(1) + "원"
        return "연회비 정보 없음"

    def save_selected_card(self, card_info, question, card_type):
        """선택된 카드 정보를 JSON 파일에 저장"""
        try:
            # 저장할 데이터 구성 (필요한 정보만)
            card_data = {
                'timestamp': datetime.now().isoformat(),
                'question': str(question),
                'card_type': str(card_type),
                'card_name': str(card_info['card_name']),
                'keyword': str(card_info['keyword'])
            }
            
            # 기존 선택된 카드 목록에 추가
            self.selected_cards.append(card_data)
            
            # JSON 파일에 저장
            filename = 'selected_cards.json'
            
            # 기존 파일이 있으면 읽어서 업데이트
            existing_cards = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        existing_cards = json.load(f)
                except json.JSONDecodeError:
                    existing_cards = []
            
            # 새로운 카드 추가
            existing_cards.append(card_data)
            
            # 파일에 저장
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_cards, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 선택하신 카드가 {filename}에 저장되었습니다.")
            return True
            
        except Exception as e:
            print(f"❌ 카드 정보 저장 중 오류 발생: {e}")
            return False
    
    def get_selected_cards(self):
        """저장된 선택 카드 목록 반환"""
        filename = 'selected_cards.json'
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []
    
    def get_card_by_name(self, card_name):
        """카드명으로 저장된 카드 정보 찾기"""
        saved_cards = self.get_selected_cards()
        for saved_card in saved_cards:
            if saved_card['card_name'] == card_name:
                return saved_card
        return None
    
    def get_recent_cards(self, limit=5):
        """최근 선택된 카드들 반환"""
        saved_cards = self.get_selected_cards()
        return saved_cards[-limit:] if saved_cards else []
    
    def clear_selected_cards(self):
        """저장된 카드 목록 삭제"""
        filename = 'selected_cards.json'
        if os.path.exists(filename):
            os.remove(filename)
            self.selected_cards = []
            print("✅ 저장된 카드 목록이 삭제되었습니다.")
        else:
            print("📋 삭제할 카드 목록이 없습니다.")
    
    def start_original_rag_chat(self, selected_card, original_question):
        """Original RAG 채팅 시작"""
        print(f"\n🎯 {selected_card['card_name']} 상세 정보 채팅")
        print("="*60)
        print("💡 이제 선택하신 카드에 대해 더 자세한 질문을 할 수 있습니다.")
        print("예시: 이용약관, 연회비 면제 조건, 할인 한도, 해외 수수료 등")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("="*60)
        
        # TODO: Original RAG 시스템 연결
        # 1. 선택된 카드의 상세 정보 로드
        # 2. 이용약관, 혜택 상세, 주의사항 등 검색
        # 3. 사용자 질문에 대한 답변 생성
        
        while True:
            try:
                chat_question = input(f"\n💬 {selected_card['card_name']}에 대해 질문하세요: ").strip()
                
                if chat_question.lower() in ['quit', 'exit', '종료']:
                    print("👋 Original RAG 채팅을 종료합니다.")
                    break
                
                if not chat_question:
                    print("❌ 질문을 입력해주세요.")
                    continue
                
                # TODO: Original RAG 처리 로직
                print(f"\n🔍 '{chat_question}' 검색 중...")
                print("📋 Original RAG 시스템에서 답변을 생성 중입니다...")
                
                # 임시 답변 (실제로는 Original RAG 시스템에서 처리)
                print(f"\n💡 {selected_card['card_name']} 관련 답변:")
                print("="*50)
                print("이 기능은 현재 개발 중입니다.")
                print("Original RAG 시스템이 연결되면 이용약관, 상세 혜택, 주의사항 등에 대한")
                print("정확한 답변을 제공할 수 있습니다.")
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n👋 Original RAG 채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 처리 중 오류가 발생했습니다: {e}")

def main():
    """실제 retriever와 연동하는 메인 함수"""
    try:
        # FAISS Retriever 임포트
        from faiss_retriever import FAISSCardRetriever
        
        print("🚀 RAG 시스템 초기화 중...")
        
        # Retriever와 Generator 초기화
        retriever = FAISSCardRetriever()
        generator = CardGenerator()
        
        print("✅ 시스템 준비 완료!")
        print("카드 타입을 먼저 선택해주세요.")
        print("1. 전체 (all)")
        print("2. 신용카드 (credit)")
        print("3. 체크카드 (check)")
        
        # 카드 타입 선택
        while True:
            card_type = input("\n카드 타입을 선택하세요 (1/2/3 또는 all/credit/check): ").strip().lower()
            
            if card_type in ['1', 'all']:
                card_type = "all"
                break
            elif card_type in ['2', 'credit', '신용카드']:
                card_type = "credit"
                break
            elif card_type in ['3', 'check', '체크카드']:
                card_type = "check"
                break
            else:
                print("❌ 올바른 옵션을 선택해주세요 (1, 2, 3 또는 all, credit, check)")
        
        card_type_display = "전체" if card_type == "all" else "신용카드" if card_type == "credit" else "체크카드"
        print(f"\n✅ 선택된 카드 타입: {card_type_display}")
        
        # 저장된 카드 목록 확인
        saved_cards = generator.get_selected_cards()
        if saved_cards:
            print(f"\n📋 이전에 선택하신 카드가 {len(saved_cards)}개 있습니다.")
            show_saved = input("저장된 카드 목록을 보시겠습니까? (y/n): ").strip().lower()
            if show_saved in ['y', 'yes', '예', '네']:
                print("\n" + "="*60)
                print("📋 저장된 카드 목록:")
                print("="*60)
                for i, saved_card in enumerate(saved_cards, 1):
                    timestamp = saved_card['timestamp'][:19]
                    print(f"{i}. {saved_card['card_name']} ({timestamp})")
                    print(f"   질문: {saved_card['question']}")
                    print(f"   유형: {saved_card['card_type']}")
                    print(f"   키워드: {saved_card['keyword']}")
                    print()
                print("="*60)
        
        print("질문을 입력하시면 추천 카드와 이유를 설명해드립니다.")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
        
        while True:
            question = input("\n💬 질문을 입력하세요: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 시스템을 종료합니다.")
                break
            
            if not question:
                print("❌ 질문을 입력해주세요.")
                continue
            
            try:
                print(f"\n🔍 '{question}' 검색 중...")
                
                # 1단계: Retriever로 검색 (상위 3개)
                search_results, _ = retriever.find_similar_cards(question, card_type, top_k=3)
                
                if not search_results:
                    print("❌ 관련 카드를 찾을 수 없습니다.")
                    continue
                
                print(f"✅ {len(search_results)}개 카드를 찾았습니다.")
                
                # 2단계: Generator로 답변 생성
                print("💬 추천 카드와 이유를 생성 중...")
                answer = generator.generate_response(question, search_results, card_type)
                
                # 3단계: 결과 출력
                print("\n" + "="*80)
                print("🎯 추천 카드와 이유:")
                print("="*80)
                print(answer)
                print("="*80)
                
                # 4단계: 비교분석 제안
                print("\n💡 3개 카드의 비교분석을 해드릴까요?")
                
                while True:
                    compare_choice = input("비교분석을 원하시면 'y', 카드 선택으로 넘어가시려면 'n'을 입력하세요: ").strip().lower()
                    
                    if compare_choice in ['y', 'yes', '예', '네']:
                        print("\n🔄 카드 비교분석을 생성 중...")
                        comparison = generator.generate_comparison(search_results)
                        print("\n" + "="*80)
                        print("📊 카드 비교분석:")
                        print("="*80)
                        print(comparison)
                        print("="*80)
                        break
                    
                    elif compare_choice in ['n', 'no', '아니오']:
                        print("✅ 카드 선택으로 넘어갑니다.")
                        break
                    
                    else:
                        print("❌ 'y' 또는 'n'을 입력해주세요.")
                
                # 5단계: 카드 선택 유도
                print("\n💡 위 카드 중 하나를 선택하시겠습니까?")
                print("선택하시면 더 자세한 정보를 제공해드립니다.")
                
                while True:
                    choice = input("\n카드를 선택하세요 (1/2/3 또는 'skip'): ").strip()
                    
                    if choice.lower() in ['skip', 's', '건너뛰기', '다음']:
                        print("✅ 다음 질문으로 넘어갑니다.")
                        break
                    
                    try:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(search_results):
                            selected_card = search_results[choice_num - 1]
                            print(f"\n{'='*60}")
                            print(f"🎉 선택하신 카드: {selected_card['card_name']}")
                            print(f"{'='*60}")
                            print(f"📋 카드 유형: {selected_card['card_type']}")
                            print(f"🔍 검색 키워드: {selected_card['keyword']}")
                            print(f"📄 상세 정보: {selected_card['card_text']}")
                            print(f"📊 유사도 점수: {selected_card['similarity_score']:.3f}")
                            print(f"{'='*60}")
                            
                            # 카드 정보 저장
                            save_success = generator.save_selected_card(selected_card, question, card_type)
                            
                            # 추가 액션 제안
                            print("\n💡 추가로 원하시는 정보가 있으신가요?")
                            print("1. 이 카드의 자세한 혜택 설명")
                            print("2. 신청 방법 안내")
                            print("3. 저장된 카드 목록 보기")
                            print("4. 다음 질문으로")
                            
                            sub_choice = input("\n선택하세요 (1/2/3/4): ").strip()
                            
                            if sub_choice == '1':
                                print(f"\n📋 {selected_card['card_name']} 상세 혜택:")
                                print("="*50)
                                # 카드 텍스트를 더 읽기 쉽게 포맷팅
                                details = selected_card['card_text'].replace(';', '\n• ').replace(',', '\n• ')
                                print(f"• {details}")
                                print("="*50)
                            
                            elif sub_choice == '2':
                                print(f"\n📞 {selected_card['card_name']} 신청 방법:")
                                print("="*50)
                                print("1. KB국민은행 홈페이지 방문")
                                print("2. KB국민은행 모바일 앱 이용")
                                print("3. KB국민은행 지점 방문")
                                print("4. 고객센터 문의: 1588-1688")
                                print("="*50)
                            
                            elif sub_choice == '3':
                                print(f"\n📋 저장된 카드 목록:")
                                print("="*50)
                                saved_cards = generator.get_selected_cards()
                                if saved_cards:
                                    for i, saved_card in enumerate(saved_cards, 1):
                                        timestamp = saved_card['timestamp'][:19]  # 날짜만 표시
                                        print(f"{i}. {saved_card['card_name']} ({timestamp})")
                                        print(f"   질문: {saved_card['question']}")
                                        print(f"   유형: {saved_card['card_type']}")
                                        print(f"   키워드: {saved_card['keyword']}")
                                        print()
                                else:
                                    print("저장된 카드가 없습니다.")
                                print("="*50)
                            
                            # 이용약관 및 상세 설명 옵션 추가
                            print(f"\n💡 {selected_card['card_name']}에 대해 더 자세히 알고 싶으신가요?")
                            print("1. 이용약관 및 상세 설명 (Original RAG 채팅)")
                            print("2. 종료")
                            
                            detail_choice = input("\n선택하세요 (1/2): ").strip()
                            
                            if detail_choice == '1':
                                print(f"\n🚀 {selected_card['card_name']} Original RAG 채팅으로 이동합니다...")
                                print("="*60)
                                print("📋 선택된 카드 정보:")
                                print(f"   카드명: {selected_card['card_name']}")
                                print(f"   키워드: {selected_card['keyword']}")
                                print(f"   원본 질문: {question}")
                                print("="*60)
                                
                                # Original RAG 채팅 시작 (향후 구현 예정)
                                print("\n🔧 Original RAG 채팅 기능은 현재 개발 중입니다.")
                                print("이 기능에서는 선택된 카드의 이용약관, 상세 혜택, 주의사항 등에 대해")
                                print("더 자세한 질문을 할 수 있습니다.")
                                print("\n예시 질문:")
                                print("- '이 카드의 이용약관을 알려줘'")
                                print("- '연회비 면제 조건이 뭐야?'")
                                print("- '할인 한도는 얼마야?'")
                                print("- '해외 사용 시 수수료는?'")
                                
                                # TODO: Original RAG 채팅 시스템 연결
                                # generator.start_original_rag_chat(selected_card, question)
                                
                            elif detail_choice == '2':
                                print("✅ 카드 선택을 완료합니다.")
                            
                            break
                        
                        else:
                            print(f"❌ 1~{len(search_results)} 사이의 숫자를 입력해주세요.")
                    
                    except ValueError:
                        print("❌ 숫자 또는 'skip'을 입력해주세요.")
                
                # 추가 질문 제안
                print("\n💡 추가 질문 예시:")
                print("- '대중교통 혜택이 좋은 카드 추천해줘'")
                print("- '연회비가 없는 카드 찾아줘'")
                print("- '쇼핑 할인 혜택이 있는 카드 알려줘'")
                
            except Exception as e:
                print(f"❌ 처리 중 오류가 발생했습니다: {e}")
    
    except ImportError:
        print("❌ faiss_retriever.py 파일을 찾을 수 없습니다.")
        print("먼저 faiss_retriever.py를 생성해주세요.")
    except Exception as e:
        print(f"❌ 초기화 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 