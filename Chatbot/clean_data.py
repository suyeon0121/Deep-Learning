import pandas as pd
import os

def clean_local_data():
    file_path = r"C:\P\Chatbot\ChatbotData (1).csv"
    
    if not os.path.exists(file_path):
        print("CSV 파일이 없습니다.")
        return

    # 1. 데이터 로드 및 중복 제거
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(['Q', 'A'])
    
    # 2. 5,000줄 추출 및 저장 (콤마 구분)
    # quoting=1 (QUOTE_ALL)을 추가하면 문장 내 콤마 에러를 더 확실히 방지합니다.
    df.head(5000)[['Q', 'A']].to_csv(
        "corpus.txt", 
        sep=',', 
        index=False, 
        header=False, 
        encoding='utf-8',
        quoting=1 
    )
            
    print(f"5,000줄의 데이터가 'corpus.txt'에 저장되었습니다.")

if __name__ == "__main__":
    clean_local_data()