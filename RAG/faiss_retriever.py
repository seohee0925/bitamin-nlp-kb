import pickle
import numpy as np
from openai import OpenAI
import os
import faiss
import time

class FAISSCardRetriever:
    def __init__(self, credit_embedding_file="../embeddings/sep_embeddings/신용카드_cards_embedding_data.pkl", 
                 check_embedding_file="../embeddings/sep_embeddings/체크카드_cards_embedding_data.pkl"):
        """FAISS 기반 카드 검색기 초기화 (신용카드/체크카드 분리)"""
        self.client = None
        
        # 신용카드와 체크카드 데이터를 별도로 저장
        self.credit_texts = None
        self.credit_metadata = None
        self.credit_faiss_index = None
        
        self.check_texts = None
        self.check_metadata = None
        self.check_faiss_index = None
        
        # 임베딩 데이터 로드
        self.load_embeddings(credit_embedding_file, check_embedding_file)
        
        # OpenAI 클라이언트 초기화
        self.init_openai_client()
    
    def load_embeddings(self, credit_file_path, check_file_path):
        """신용카드와 체크카드 임베딩 데이터 로드"""
        try:
            # 신용카드 데이터 로드
            if os.path.exists(credit_file_path):
                with open(credit_file_path, 'rb') as f:
                    credit_data = pickle.load(f)
                
                self.credit_texts = credit_data['texts']
                self.credit_metadata = credit_data['metadata']
                self.credit_faiss_index = credit_data['faiss_index']
            else:
                print(f"⚠️  신용카드 임베딩 파일을 찾을 수 없습니다: {credit_file_path}")
            
            # 체크카드 데이터 로드
            if os.path.exists(check_file_path):
                with open(check_file_path, 'rb') as f:
                    check_data = pickle.load(f)
                
                self.check_texts = check_data['texts']
                self.check_metadata = check_data['metadata']
                self.check_faiss_index = check_data['faiss_index']
            else:
                print(f"⚠️  체크카드 임베딩 파일을 찾을 수 없습니다: {check_file_path}")
            
            # 최소한 하나의 데이터는 있어야 함
            if self.credit_texts is None and self.check_texts is None:
                raise FileNotFoundError("신용카드와 체크카드 임베딩 파일이 모두 없습니다.")
            
        except Exception as e:
            print(f"❌ 임베딩 파일 로드 중 오류가 발생했습니다: {e}")
            print("먼저 embed_cards.py를 실행하여 FAISS 임베딩 데이터를 생성해주세요.")
            raise
    
    def init_openai_client(self):
        """OpenAI 클라이언트 초기화"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = input("OpenAI API 키를 입력하세요: ")
        
        self.client = OpenAI(api_key=api_key)
    
    def get_question_embedding(self, question):
        """질문을 벡터로 변환"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
        return response.data[0].embedding
    
    def find_similar_cards(self, question, card_type="all", top_k=5):
        """FAISS를 사용하여 질문과 가장 유사한 카드들을 찾기 (코사인 유사도 사용)"""
        
        start_time = time.time()
        
        # 질문을 벡터로 변환
        question_embedding = self.get_question_embedding(question)
        question_vector = np.array(question_embedding).astype('float32').reshape(1, -1)
        
        # 질문 벡터 정규화 (코사인 유사도 계산을 위해)
        question_norm = np.linalg.norm(question_vector)
        if question_norm > 0:
            question_vector = question_vector / question_norm
        
        all_results = []
        
        # 카드 타입에 따라 검색
        if card_type.lower() in ["all", "credit", "신용카드"] and self.credit_faiss_index:
            # 신용카드 검색
            distances, indices = self.credit_faiss_index.search(question_vector, top_k)
            
            for i, (similarity, idx) in enumerate(zip(distances[0], indices[0])):
                # FAISS의 IndexFlatIP는 내적을 반환하므로 직접 코사인 유사도
                # 벡터가 정규화되어 있으므로 내적 = 코사인 유사도
                cosine_similarity = similarity
                card_meta = self.credit_metadata[idx]
                card_text = self.credit_texts[idx]
                
                all_results.append({
                    'rank': len(all_results) + 1,
                    'card_name': card_meta['card_name'],
                    'card_type': card_meta['card_type'],
                    'keyword': card_meta['keyword'],
                    'similarity_score': round(cosine_similarity, 4),
                    'distance': round(1 - cosine_similarity, 4),  # 코사인 거리로 변환
                    'card_text': card_text,
                    'search_type': '신용카드'
                })
        
        if card_type.lower() in ["all", "check", "체크카드"] and self.check_faiss_index:
            # 체크카드 검색
            distances, indices = self.check_faiss_index.search(question_vector, top_k)
            
            for i, (similarity, idx) in enumerate(zip(distances[0], indices[0])):
                # FAISS의 IndexFlatIP는 내적을 반환하므로 직접 코사인 유사도
                # 벡터가 정규화되어 있으므로 내적 = 코사인 유사도
                cosine_similarity = similarity
                card_meta = self.check_metadata[idx]
                card_text = self.check_texts[idx]
                
                all_results.append({
                    'rank': len(all_results) + 1,
                    'card_name': card_meta['card_name'],
                    'card_type': card_meta['card_type'],
                    'keyword': card_meta['keyword'],
                    'similarity_score': round(cosine_similarity, 4),
                    'distance': round(1 - cosine_similarity, 4),  # 코사인 거리로 변환
                    'card_text': card_text,
                    'search_type': '체크카드'
                })
        
        # 유사도 점수로 정렬 (코사인 유사도는 높을수록 유사)
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # 순위 재정렬
        for i, result in enumerate(all_results):
            result['rank'] = i + 1
        
        search_time = time.time() - start_time
        
        return all_results[:top_k], search_time
    
    def search_cards(self, question, card_type="all", top_k=5):
        """카드 검색 실행"""
        card_type_display = "전체" if card_type.lower() == "all" else card_type
        print(f"🔍 '{question}' 검색 결과 (카드타입: {card_type_display})")
        print("=" * 60)
        
        results, _ = self.find_similar_cards(question, card_type, top_k)
        
        for result in results:
            print(f"\n📋 카드명: {result['card_name']}")
            print(f"🏷️  유형: {result['card_type']}")
            print(f"🔑 키워드: {result['keyword']}")
            print(f"📝 상세정보:")
            print(f"   {result['card_text']}")
            print("-" * 60)
        
        return results
    
    def batch_search(self, questions, card_type="all", top_k=3):
        """여러 질문을 한번에 검색"""
        card_type_display = "전체" if card_type.lower() == "all" else card_type
        print(f"🔄 {len(questions)}개 질문을 배치 검색합니다... (카드타입: {card_type_display})\n")
        
        all_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"질문 {i}/{len(questions)}: {question}")
            results, _ = self.find_similar_cards(question, card_type, top_k)
            
            all_results.append({
                'question': question,
                'results': results
            })
            
            # 상위 1개 결과만 간단히 출력
            if results:
                top_result = results[0]
                print(f"  → {top_result['card_name']}")
            print()
        
        print(f"✅ 배치 검색 완료!")
        return all_results

def main():
    """메인 함수 - FAISS 기반 대화형 검색"""
    try:
        retriever = FAISSCardRetriever()
        
        print("\n🚀 FAISS 기반 카드 검색기가 준비되었습니다!")
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
        print("질문을 입력하시면 관련 카드를 찾아드립니다.")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("배치 검색을 원하면 'batch'를 입력하세요.\n")
        
        while True:
            question = input("\n💬 질문을 입력하세요: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 검색을 종료합니다.")
                break
            
            if question.lower() == 'batch':
                # 배치 검색 모드
                print("\n📝 배치 검색 모드입니다.")
                print("질문들을 한 줄씩 입력하세요. 빈 줄을 입력하면 검색을 시작합니다.")
                
                questions = []
                while True:
                    q = input("질문: ").strip()
                    if not q:
                        break
                    questions.append(q)
                
                if questions:
                    retriever.batch_search(questions, card_type, top_k=3)
                continue
            
            if not question:
                print("❌ 질문을 입력해주세요.")
                continue
            
            try:
                results = retriever.search_cards(question, card_type, top_k=3)
                
                # 추가 질문 제안
                print("\n💡 추가 질문 예시:")
                print("- '대중교통 혜택이 좋은 카드 추천해줘'")
                print("- '연회비가 없는 카드 찾아줘'")
                print("- '쇼핑 할인 혜택이 있는 카드 알려줘'")
                print("- 'batch' 입력으로 여러 질문 한번에 검색")
                
            except Exception as e:
                print(f"❌ 검색 중 오류가 발생했습니다: {e}")
    
    except Exception as e:
        print(f"❌ 초기화 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 