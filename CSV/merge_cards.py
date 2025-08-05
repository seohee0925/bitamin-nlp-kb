import os
import json
import glob

def merge_all_card_jsons(input_folders, output_file):
    card_list = []
    for input_folder in input_folders:
        # 하위 폴더까지 모든 json 파일 찾기
        json_files = glob.glob(os.path.join(input_folder, "**", "*.json"), recursive=True)
        print(f"{input_folder}: {len(json_files)}개 파일을 병합합니다.")
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    card_list.extend(data)
                else:
                    card_list.append(data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(card_list, f, ensure_ascii=False, indent=2)
    print(f"완료! {output_file}에 {len(card_list)}개 카드가 저장되었습니다.")

if __name__ == "__main__":
    # 신용_summary, 체크_summary 폴더 전체 병합
    merge_all_card_jsons(["신용_summary", "체크_summary"], "cards_summary.json")