import json
import pandas as pd
import os

def add_intro_to_json():
    """card_summary.json에 KB카드_전체통합.csv의 소개 정보를 추가"""
    
    # KB카드_전체통합.csv 파일 읽기
    csv_file = "KB카드_전체통합.csv"
    if not os.path.exists(csv_file):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")
        return
    
    print(f"📄 {csv_file} 파일을 읽는 중...")
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 카드명, 카드타입, 소개를 딕셔너리로 변환
    card_info = {}
    for _, row in df.iterrows():
        # NaN 값 처리
        card_name = str(row['카드명']).strip() if pd.notna(row['카드명']) else ""
        card_type = str(row['카드타입']).strip() if pd.notna(row['카드타입']) else ""
        introduction = str(row['소개']).strip() if pd.notna(row['소개']) else ""
        
        # 빈 값이 아닌 경우만 추가
        if card_name and card_type and introduction:
            key = f"{card_name}_{card_type}"
            card_info[key] = introduction
    
    print(f"✅ {len(card_info)}개 카드의 정보를 로드했습니다.")
    
    # card_summary.json 파일 읽기
    json_file = "cards_summary.json"
    if not os.path.exists(json_file):
        print(f"❌ 파일을 찾을 수 없습니다: {json_file}")
        return
    
    print(f"📄 {json_file} 파일을 읽는 중...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)
    
    print(f"✅ {len(cards_data)}개 카드를 읽었습니다.")
    
    # 각 카드에 intro 필드 추가
    matched_count = 0
    unmatched_count = 0
    
    for card in cards_data:
        card_name = card.get('card_name', '').strip()
        card_type = card.get('card_type', '').strip()
        
        if card_name and card_type:
            # 카드타입 매핑 (신용카드 -> 신용카드, 체크카드 -> 체크카드)
            mapped_card_type = card_type
            
            key = f"{card_name}_{mapped_card_type}"
            
            if key in card_info:
                card['intro'] = card_info[key]
                matched_count += 1
            else:
                unmatched_count += 1
        else:
            unmatched_count += 1
    
    # 결과 저장
    output_file = "card_summary_with_intro.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cards_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 완료!")
    print(f"📁 저장 파일: {output_file}")
    print(f"📊 매칭된 카드: {matched_count}개")
    print(f"📊 매칭되지 않은 카드: {unmatched_count}개")
    print(f"📊 총 카드 수: {len(cards_data)}개")
    
    # 매칭 통계
    if matched_count > 0:
        match_rate = (matched_count / len(cards_data)) * 100
        print(f"📈 매칭률: {match_rate:.1f}%")
    
    # 샘플 출력
    print(f"\n🔍 샘플 결과 (처음 3개 카드):")
    for i, card in enumerate(cards_data[:3]):
        print(f"\n{i+1}. {card.get('card_name', 'N/A')}")
        print(f"   타입: {card.get('card_type', 'N/A')}")
        print(f"   소개: {card.get('intro', '소개 없음')}")
    
    return {
        'matched_count': matched_count,
        'unmatched_count': unmatched_count,
        'total_cards': len(cards_data),
        'output_file': output_file
    }

def main():
    """메인 함수"""
    print("🚀 JSON 파일에 소개 정보 추가 시작")
    print("=" * 50)
    
    result = add_intro_to_json()
    
    if result:
        print("\n" + "=" * 50)
        print("🎉 소개 정보 추가 작업이 완료되었습니다!")

if __name__ == "__main__":
    main() 