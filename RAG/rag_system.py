import pickle
import numpy as np
from openai import OpenAI
import os
import faiss
import time
from card_generator import CardGenerator

class RAGCardSystem:
    def __init__(self, embedding_file="embeddings/cards_embedding_data.pkl"):
        """RAG 카드 시스템 초기화"""
        self.client = None
        self.texts = None
        self.metadata = None
        self.faiss_index = None
        self.generator = None
        
        # 임베딩 데이터 로드
        self.load_embeddings(embedding_file)
        
        # OpenAI 클라이언트 초기화
        self.init_openai_client()
        
        # Generator 초기화
        self.generator = CardGenerator()
    
    def load_embeddings(self, file_path):
        """임베딩 데이터 로드"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.texts = data['texts']
            self.metadata = data['metadata']
            self.faiss_index = data['faiss_index']
            
            print(f"✅ {len(self.texts)}개 카드 데이터를 로드했습니다.")
            print(f"🔍 FAISS 인덱스 크기: {self.faiss_index.ntotal}개 벡터")
            
        except FileNotFoundError:
            print(f"❌ 임베딩 파일을 찾을 수 없습니다: {file_path}")
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
    
    def retrieve_cards(self, question, top_k=5):
        """FAISS를 사용하여 관련 카드 검색"""
        
        start_time = time.time()
        
        # 질문을 벡터로 변환
        question_embedding = self.get_question_embedding(question)
        question_vector = np.array(question_embedding).astype('float32').reshape(1, -1)
        
        # FAISS로 유사한 벡터 검색
        distances, indices = self.faiss_index.search(question_vector, top_k)
        
        search_time = time.time() - start_time
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            similarity_score = 1 / (1 + distance)
            
            card_meta = self.metadata[idx]
            card_text = self.texts[idx]
            
            results.append({
                'rank': i + 1,
                'card_name': card_meta['card_name'],
                'card_type': card_meta['card_type'],
                'keyword': card_meta['keyword'],
                'similarity_score': round(similarity_score, 4),
                'distance': round(distance, 4),
                'card_text': card_text
            })
        
        return results, search_time
    
    def generate_answer(self, question, search_results):
        """검색 결과를 바탕으로 답변 생성"""
        return self.generator.generate_response(question, search_results)
    
    def process_question(self, question, top_k=3):
        """질문 처리: 검색 + 답변 생성"""
        
        print(f"🔍 질문: '{question}'")
        print("=" * 80)
        
        # 1단계: 검색 (Retrieval)
        print("📊 관련 카드 검색 중...")
        search_results, search_time = self.retrieve_cards(question, top_k)
        
        print(f"⚡ 검색 완료! ({search_time:.4f}초)")
        print(f"📋 {len(search_results)}개 카드를 찾았습니다.\n")
        
        # 검색 결과 출력
        print("🔍 검색된 카드들:")
        for result in search_results:
            print(f"  {result['rank']}위: {result['card_name']} (유사도: {result['similarity_score']})")
        print()
        
        # 2단계: 답변 생성 (Generation)
        print("💬 답변 생성 중...")
        start_time = time.time()
        answer = self.generate_answer(question, search_results)
        generation_time = time.time() - start_time
        
        print(f"✅ 답변 생성 완료! ({generation_time:.4f}초)\n")
        
        # 최종 답변 출력
        print("🎯 최종 답변:")
        print("=" * 80)
        print(answer)
        print("=" * 80)
        
        # 성능 요약
        total_time = search_time + generation_time
        print(f"\n📈 성능 요약:")
        print(f"  - 검색 시간: {search_time:.4f}초")
        print(f"  - 생성 시간: {generation_time:.4f}초")
        print(f"  - 총 시간: {total_time:.4f}초")
        
        return {
            'question': question,
            'search_results': search_results,
            'answer': answer,
            'search_time': search_time,
            'generation_time': generation_time,
            'total_time': total_time
        }

def main():
    """메인 함수 - RAG 시스템 실행"""
    try:
        rag_system = RAGCardSystem()
        
        print("\n🚀 RAG 카드 상담 시스템이 준비되었습니다!")
        print("질문을 입력하시면 관련 카드를 검색하고 답변을 생성합니다.")
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
                result = rag_system.process_question(question, top_k=3)
                
                # 추가 질문 제안
                print("\n💡 추가 질문 예시:")
                print("- '대중교통 혜택이 좋은 카드 추천해줘'")
                print("- '연회비가 없는 카드 찾아줘'")
                print("- '쇼핑 할인 혜택이 있는 카드 알려줘'")
                print("- '학생증 카드 추천해줘'")
                
            except Exception as e:
                print(f"❌ 처리 중 오류가 발생했습니다: {e}")
    
    except Exception as e:
        print(f"❌ 초기화 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 