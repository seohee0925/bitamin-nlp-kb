import json
import os
import sys
from openai import OpenAI

def load_prompt():
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        return f.read()

def summarize_card(api_key, card_json):
    client = OpenAI(api_key=api_key)
    prompt = load_prompt().replace("{input_json}", json.dumps(card_json, ensure_ascii=False, indent=2))
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "카드 정보를 분석하고 요약하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    
    summary_text = response.choices[0].message.content.strip()
    
    if "```json" in summary_text:
        start_idx = summary_text.find("```json") + 7
        end_idx = summary_text.find("```", start_idx)
        if end_idx == -1:
            end_idx = len(summary_text)
        json_str = summary_text[start_idx:end_idx].strip()
    else:
        json_str = summary_text
    
    return json.loads(json_str)

def process_file(api_key, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        card_data = json.load(f)
    
    summary = summarize_card(api_key, card_data)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

def main():
    # 체크json/ 폴더의 모든 JSON 파일 처리
    input_folder = "신용json/통신"
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("OpenAI API 키를 입력하세요: ")
    
    if not os.path.exists(input_folder):
        print(f"폴더를 찾을 수 없습니다: {input_folder}")
        sys.exit(1)
    
    # 모든 JSON 파일 찾기
    import glob
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print(f"JSON 파일을 찾을 수 없습니다: {input_folder}")
        sys.exit(1)
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    for i, file_path in enumerate(json_files, 1):
        try:
            print(f"\n=== {i}/{len(json_files)} 처리 중 ===")
            print(f"파일: {file_path}")
            
            # 파일명만 추출
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            # 체크_summary/ 폴더에 저장
            output_dir = "신용_summary/통신"
            output_file = os.path.join(output_dir, f"{base_name}_summary.json")
            
            # 이미 처리된 파일인지 확인
            if os.path.exists(output_file):
                print(f"이미 존재하는 파일입니다: {output_file}")
                continue
            
            summary = process_file(api_key, file_path, output_file)
            
            if summary:
                print(f"✅ 완료! 카드명: {summary.get('card_name', 'N/A')}")
                print(f"📁 저장 위치: {output_file}")
            else:
                print("❌ 실패!")
                
            # API Rate Limiting 방지를 위한 대기
            if i < len(json_files):  # 마지막 파일이 아니면 대기
                print("⏳ 3초 대기 중... (API Rate Limiting 방지)")
                import time
                time.sleep(3)
                
        except Exception as e:
            print(f"❌ 오류 발생 ({file_path}): {e}")
            print("다음 파일로 진행합니다...")
            continue
    
    print(f"\n모든 파일 처리 완료!")

if __name__ == "__main__":
    main()
