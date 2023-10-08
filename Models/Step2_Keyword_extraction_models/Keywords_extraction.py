# 필요 라이브러리 선언
import pandas as pd
import spacy
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


import os
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

# KeyBERT 모델 생성 및 Pre-train model 객체를 내부에서 생성
kw_model = KeyBERT('all-mpnet-base-v2')
nlp = spacy.load('en_core_web_lg')


# Init vectorizer for the English language
# '<ADJ.*>*<N.*>+' extracts keywords that have 0 or more adjectives, followed by 1 or more nouns using the English spaCy part-of-speech tags.
vectorizer_1 = KeyphraseCountVectorizer(spacy_pipeline=nlp, pos_pattern='<ADJ>*<N.*>+', stop_words='english') # 형용사 + 최소 하나 이상의 명사
vectorizer_2 = KeyphraseCountVectorizer(spacy_pipeline=nlp, pos_pattern='(<N.*>+<N.*>+)', stop_words='english') # 명사 + 명사
vectorizer_3 = KeyphraseCountVectorizer(spacy_pipeline=nlp, pos_pattern='<JJ.*>+<NN.*>', stop_words='english') # 형용사 + 명사


# 사용자 정의 vectorizer 사용
def extract_keywords_for_beer_2(doc):
    try:
        keywords = kw_model.extract_keywords(doc, vectorizer=vectorizer_3, top_n=5)
    except ValueError:
        # 키워드를 찾지 못한 경우 빈 리스트 반환
        keywords = []
    return [keyword for keyword, score in keywords]


# 데이터 불러오기
df = pd.read_csv('../../Data/Preprocessed_data/pp_selected_reviews.csv')


# # 리뷰 데이터 1개에 대해 키워드 추출 함수 적용
# print("[리뷰]")
# print(df['Review'].iloc[0])
# k = extract_keywords_for_beer_2(df['Review'].iloc[0])
# print("\n[추출 키워드: 사용자 정의 vectorizer]")
# print(*k, sep="\n")


# 데이터셋에 키워드 추출 함수 적용 및 csv파일 생성
df['Keywords'] = df['Review'].apply(extract_keywords_for_beer_2)
df.to_csv('keyJJNN_pp_reviews.csv', encoding='utf-8', index = False)
