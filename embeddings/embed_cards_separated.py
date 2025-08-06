import json
import os
import numpy as np
from openai import OpenAI
import pickle
import faiss

def json_to_text(card_data):
    """카드 JSON을 텍스트로 변환"""
    text_parts = []
    
    # 기본 정보
    text_parts.append(f"카드명: {card_data.get('card_name', 'N/A')}")
    text_parts.append(f"카드유형: {card_data.get('card_type', 'N/A')}")
    text_parts.append(f"키워드: {card_data.get('keyword', 'N/A')}")
    text_parts.append(f"브랜드: {card_data.get('brand', 'N/A')}")
    text_parts.append(f"발급대상: {card_data.get('target_user', 'N/A')}")
    
    # 소개 (intro) 추가
    intro = card_data.get('intro', '')
    if intro:
        text_parts.append(f"소개: {intro}")
    
    # 혜택
    benefits = card_data.get('benefits', [])
    if benefits:
        benefits_text = ", ".join(benefits)
        text_parts.append(f"혜택: {benefits_text}")
    
    # 혜택 조건
    conditions = card_data.get('benefit_conditions', [])
    if conditions:
        conditions_text = ", ".join(conditions)
        text_parts.append(f"조건: {conditions_text}")
    
    # 연회비
    fee = card_data.get('fee', 'N/A')
    text_parts.append(f"연회비: {fee}")
    
    # 출시일
    release_date = card_data.get('release_date', '')
    if release_date:
        text_parts.append(f"출시일: {release_date}")
    
    return " | ".join(text_parts)

def get_embedding(client, text):
    """텍스트를 벡터로 변환"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def process_cards_by_type(cards_data, card_type, client, output_dir):
    """특정 타입의 카드들을 처리하여 임베딩 생성"""
    
    # 해당 타입의 카드만 필터링
    filtered_cards = [card for card in cards_data if card.get('card_type', '').lower() == card_type.lower()]
    
    if not filtered_cards:
        print(f"⚠️  {card_type} 타입의 카드가 없습니다.")
        return
    
    print(f"📊 {card_type} 카드 {len(filtered_cards)}개를 처리합니다.")
    
    # 결과 저장용 리스트
    card_texts = []
    card_embeddings = []
    card_metadata = []
    
    for i, card in enumerate(filtered_cards, 1):
        try:
            print(f"처리 중: {i}/{len(filtered_cards)} - {card.get('card_name', 'N/A')}")
            
            # JSON을 텍스트로 변환
            card_text = json_to_text(card)
            card_texts.append(card_text)
            
            # 벡터화
            embedding = get_embedding(client, card_text)
            card_embeddings.append(embedding)
            
            # 메타데이터 저장
            metadata = {
                'card_name': card.get('card_name', 'N/A'),
                'card_type': card.get('card_type', 'N/A'),
                'keyword': card.get('keyword', 'N/A'),
                'index': i-1
            }
            card_metadata.append(metadata)
            
        except Exception as e:
            print(f"오류 발생 ({card.get('card_name', 'N/A')}): {e}")
            continue
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 텍스트 저장
    text_filename = f"{card_type.lower()}_card_texts.txt"
    with open(os.path.join(output_dir, text_filename), 'w', encoding='utf-8') as f:
        for i, text in enumerate(card_texts):
            f.write(f"[{i+1}] {text}\n\n")
    
    # 벡터를 numpy 배열로 변환
    embeddings_array = np.array(card_embeddings).astype('float32')
    
    # 벡터 정규화 (코사인 유사도 계산을 위해)
    faiss.normalize_L2(embeddings_array)
    
    # FAISS 인덱스 생성 (코사인 유사도 기반 - 내적 사용)
    dimension = embeddings_array.shape[1]  # 벡터 차원
    index = faiss.IndexFlatIP(dimension)  # Inner Product (내적) 사용
    
    # 벡터를 FAISS 인덱스에 추가
    index.add(embeddings_array)
    
    # FAISS 인덱스 저장
    faiss_filename = f"{card_type.lower()}_card_embeddings.faiss"
    faiss.write_index(index, os.path.join(output_dir, faiss_filename))
    
    # 메타데이터 저장
    metadata_filename = f"{card_type.lower()}_card_metadata.json"
    with open(os.path.join(output_dir, metadata_filename), 'w', encoding='utf-8') as f:
        json.dump(card_metadata, f, ensure_ascii=False, indent=2)
    
    # 모든 데이터를 하나의 파일로 저장 (pickle)
    all_data = {
        'texts': card_texts,
        'embeddings': embeddings_array,
        'metadata': card_metadata,
        'faiss_index': index
    }
    pkl_filename = f"{card_type.lower()}_cards_embedding_data.pkl"
    with open(os.path.join(output_dir, pkl_filename), 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"\n✅ {card_type} 카드 처리 완료!")
    print(f"📄 텍스트 파일: {output_dir}/{text_filename}")
    print(f"🔍 FAISS 인덱스: {output_dir}/{faiss_filename}")
    print(f"📋 메타데이터: {output_dir}/{metadata_filename}")
    print(f"💾 통합 파일: {output_dir}/{pkl_filename}")
    print(f"📊 총 {len(card_texts)}개 {card_type} 카드가 처리되었습니다.")
    print(f"🔢 FAISS 인덱스 크기: {index.ntotal}개 벡터")
    print(f"📏 벡터 차원: {index.d}")

def process_cards_to_embeddings_separated(input_file, output_dir):
    """카드 JSON을 카드 타입별로 나누어 텍스트로 변환하고 벡터화"""
    
    # OpenAI 클라이언트 초기화
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("OpenAI API 키를 입력하세요: ")
    
    client = OpenAI(api_key=api_key)
    
    # JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    
    print(f"총 {len(cards_data)}개 카드를 카드 타입별로 분리하여 처리합니다.")
    
    # 카드 타입별로 분리
    card_types = {}
    for card in cards_data:
        card_type = card.get('card_type', 'Unknown')
        if card_type not in card_types:
            card_types[card_type] = []
        card_types[card_type].append(card)
    
    print(f"발견된 카드 타입: {list(card_types.keys())}")
    
    # 각 카드 타입별로 처리
    for card_type, cards in card_types.items():
        print(f"\n{'='*50}")
        print(f"🔍 {card_type} 카드 처리 시작")
        print(f"{'='*50}")
        process_cards_by_type(cards_data, card_type, client, output_dir)
    
    print(f"\n🎉 모든 카드 타입별 처리 완료!")
    print(f"📁 결과 파일들이 {output_dir} 디렉토리에 저장되었습니다.")

def main():
    """메인 함수"""
    # 입력 파일과 출력 디렉토리 설정
    input_file = "cards_summary_with_intro.json"
    output_dir = "embeddings"
    
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        print("cards_summary_with_intro.json 파일이 현재 디렉토리에 있는지 확인해주세요.")
        return
    
    process_cards_to_embeddings_separated(input_file, output_dir)

if __name__ == "__main__":
    main() 