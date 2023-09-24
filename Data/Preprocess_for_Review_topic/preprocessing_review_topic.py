import os
import re
import pandas as pd

# GPU 사용 환경 구성
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 텍스트 정규화 수행: 같은 의미지만 다른 표현으로 쓰이는 경우가 있기에, 연산량을 줄이고자 정규화 과정이 필요
contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have"}

print("정규화 사전의 수: ", len(contractions))

def preprocess_sentence(sentence):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = re.sub(r'\d+ml|\d+ ml', '', sentence)  # "ml"을 포함하는 문자 제거
    sentence = re.sub(r'\d+oz|\d+ oz', '', sentence)  # "oz"을 포함하는 문자 제거
    sentence = re.sub(r'\d+cl|\d+ cl', '', sentence)  # "cl"을 포함하는 문자 제거
    sentence = re.sub(r'---rated via beer buddy for iphone|---rated via beerbuddy for iphone', '', sentence)  # 특정 문구 제거
    sentence = re.sub(r"[\d\/\d\/\d]", "", sentence) # 날짜 삭제
    sentence = re.sub(r"bottle[ .,?!:]|bottled[ .,?!:]", "", sentence) # bottle 관련 삭제
    sentence = re.sub(r'on tap[ .,?!]|tap[ .,?!]', '', sentence)  # "on tap, tap"을 포함하는 문자 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (.) 제거
    sentence = re.sub(r'[.,?!]+[/.,?!]', '', sentence) # 여러개 자음, 모음, 구두점 제거
    sentence = re.sub(r"[^a-z0-9.,!?'"":]", " ", sentence) # 지정한 문자 제외 공백으로 전환
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r'[" "]+', " ", sentence) # 여러개 공백을 하나의 공백으로 바꿈
    sentence = sentence.strip() # 문장 양쪽 공백 제거
    return sentence
# 수정 및 추가 사항
# on tap[특수기호] 형태 다수 포착하여 처리
# bottle[특수기호] 형태 다수 포착하여 처리 (+bottled 필터 추가)
# 카테고리별로 리뷰한 데이터에 ':'가 삭제되어 카테고리+리뷰가 나옴 
# (원문 Appearance: pitch black, Flavour: biscuit, coffee, ...)
# (추출 appearance pitch black, flavour biscuit)


# 감정 분석 데이터 불러오기
df = pd.read_csv('../Data/Binary Classification_v4.csv')


# 해당하는 맥주 이름을 가진 리뷰들만 선택하기
beer_names = ['Asahi Super Dry', '8 Wired iStout', 'Red Rock']
selected_reviews = df[df['Beer_name'].isin(beer_names)]
selected_reviews.reset_index(drop=True, inplace=True)


# 지정된 맥주 리뷰 데이터프레임 전처리 적용
processed_reviews = selected_reviews.copy()
processed_reviews['Review'] = processed_reviews['Review'].apply(lambda x: preprocess_sentence(x))


# # 전체 맥주 리뷰 데이터프레임 전처리 적용
# processed_reviews = df.copy()
# processed_reviews['Review'] = processed_reviews['Review'].apply(lambda x: preprocess_sentence(x))


# 전처리된 데이터프레임 csv 파일 생성
processed_reviews.to_csv('../Data/pp_selected_reviews.csv', index=False)
processed_reviews.info()