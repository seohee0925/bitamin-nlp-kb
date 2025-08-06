import json
import re
import os

def parse_card_text(text_line):
    """카드 텍스트를 파싱하여 JSON 구조로 변환"""
    
    # [숫자] 패턴 제거
    text_line = re.sub(r'^\[\d+\]\s*', '', text_line.strip())
    
    # 파이프(|)로 구분된 필드들을 분리
    fields = text_line.split(' | ')
    
    card_data = {}
    
    for field in fields:
        if ':' not in field:
            continue
            
        key, value = field.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        if key == '카드명':
            card_data['card_name'] = value
        elif key == '카드유형':
            card_data['card_type'] = value
        elif key == '키워드':
            card_data['keyword'] = value
        elif key == '브랜드':
            # 브랜드는 쉼표로 구분된 리스트로 변환
            brands = [brand.strip() for brand in value.split(',')]
            card_data['brand'] = brands
        elif key == '발급대상':
            card_data['target_user'] = value
        elif key == '소개':
            card_data['intro'] = value
        elif key == '혜택':
            # 혜택은 쉼표로 구분된 리스트로 변환
            benefits = [benefit.strip() for benefit in value.split(',')]
            card_data['benefits'] = benefits
        elif key == '조건':
            # 조건은 쉼표로 구분된 리스트로 변환
            conditions = [condition.strip() for condition in value.split(',')]
            card_data['benefit_conditions'] = conditions
        elif key == '연회비':
            card_data['fee'] = value
        elif key == '출시일':
            card_data['release_date'] = value
    
    return card_data

def read_card_texts_file(file_path):
    """카드 텍스트 파일을 읽어서 카드 데이터 리스트로 변환"""
    cards = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_card_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            # 빈 줄이면 현재 카드 텍스트를 파싱
            if current_card_text:
                card_data = parse_card_text(current_card_text)
                if card_data.get('card_name'):  # 카드명이 있는 경우만 추가
                    cards.append(card_data)
                current_card_text = ""
        else:
            # 빈 줄이 아니면 현재 카드 텍스트에 추가
            if current_card_text:
                current_card_text += " | " + line
            else:
                current_card_text = line
    
    # 마지막 카드 처리
    if current_card_text:
        card_data = parse_card_text(current_card_text)
        if card_data.get('card_name'):
            cards.append(card_data)
    
    return cards

def restore_json_from_texts():
    """텍스트 파일들에서 JSON 파일을 복구"""
    
    # 파일 경로 설정
    credit_file = "sep_embeddings/신용카드_card_texts.txt"
    check_file = "sep_embeddings/체크카드_card_texts.txt"
    output_file = "cards_summary_with_intro.json"
    
    all_cards = []
    
    # 신용카드 데이터 읽기
    if os.path.exists(credit_file):
        print(f"📖 {credit_file} 파일을 읽는 중...")
        credit_cards = read_card_texts_file(credit_file)
        print(f"✅ {len(credit_cards)}개의 신용카드 데이터를 읽었습니다.")
        all_cards.extend(credit_cards)
    else:
        print(f"⚠️  {credit_file} 파일을 찾을 수 없습니다.")
    
    # 체크카드 데이터 읽기
    if os.path.exists(check_file):
        print(f"📖 {check_file} 파일을 읽는 중...")
        check_cards = read_card_texts_file(check_file)
        print(f"✅ {len(check_cards)}개의 체크카드 데이터를 읽었습니다.")
        all_cards.extend(check_cards)
    else:
        print(f"⚠️  {check_file} 파일을 찾을 수 없습니다.")
    
    if not all_cards:
        print("❌ 읽을 수 있는 카드 데이터가 없습니다.")
        return
    
    # JSON 파일로 저장
    print(f"💾 {len(all_cards)}개의 카드 데이터를 {output_file}에 저장하는 중...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cards, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {output_file} 파일이 성공적으로 생성되었습니다!")
    
    # 통계 출력
    card_types = {}
    keywords = {}
    
    for card in all_cards:
        card_type = card.get('card_type', 'Unknown')
        keyword = card.get('keyword', 'Unknown')
        
        card_types[card_type] = card_types.get(card_type, 0) + 1
        keywords[keyword] = keywords.get(keyword, 0) + 1
    
    print(f"\n📊 복구된 데이터 통계:")
    print(f"총 카드 수: {len(all_cards)}개")
    print(f"카드 타입별:")
    for card_type, count in sorted(card_types.items()):
        print(f"  - {card_type}: {count}개")
    print(f"키워드별:")
    for keyword, count in sorted(keywords.items()):
        print(f"  - {keyword}: {count}개")

if __name__ == "__main__":
    restore_json_from_texts() 