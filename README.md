# Project : Beer Sentiment Classifier and Keyphrase Extraction
<img src="https://img.shields.io/badge/Python-3.8-blue"><img src="https://img.shields.io/badge/Transformers-4.16.2-blue"><img src="https://img.shields.io/badge/-Colab-yellow)"><img src="https://img.shields.io/badge/Pytorch-blue">

## 프로젝트 소개
- 본 프로젝트는 리뷰 데이터 감정분석과 핵심 문구를 추출하는 방법입니다.
- 데이터 수집, 라벨링, 모델링, 데이터 증강, 문구 추출방법을 전부 다룹니다.

## 프로젝트 목표
### 모델의 감정 분석 성능을 Precision 기준 0.9 이상 달성
### 사용자가 선택한 맥주에 대한 핵심 문구을 추출하는 방법 제공

## 감정 분류 및 핵심 문구 추출 Demo

## 사전 준비
``` c 
!git clone https://github.com/sgr1118/Beer_Sentiment_analysis.git
pip install -r requirements.txt
```

## 사용법 예시
### Use to Pre_Trained_Model [[colab]](https://colab.research.google.com/drive/1JhGI6jTBXHxkXtQKYtA__V0kQYu1mlTk#scrollTo=tuOrfo06qbsv)

``` c 
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

# 모델 로드
model = BertForSequenceClassification.from_pretrained('GiRak/beer-sentiment-bert') # HuggingFace 사전 학습 모델 업로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 토크나이저 초기화
tokenizer = BertTokenizerFast.from_pretrained('GiRak/beer-sentiment-bert') # HuggingFace 사전 학습 모델 업로드

def analyze_sentiment(sentence):
    # 문장을 토크나이징하고 모델 입력으로 변환
    inputs = tokenizer(sentence, return_tensors='pt')
    inputs = inputs.to(device)

    # 모델을 통해 감정 분류 수행
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)

    # 감정 분류 확률 추출
    sentiment_labels = ['Negative', 'Positive']
    sentiment_probabilities = {label: probability.item() for label, probability in zip(sentiment_labels, probabilities[0])}

    return sentiment_probabilities

sentences = ['I took a sip and immediately discarded it. How could a beer have such a strong cinnamon flavor?']

# Lists to store probabilities
positive_probs = []
negative_probs = []

for sentence in sentences:
    sentiment_probabilities = analyze_sentiment(sentence)
    positive_prob = sentiment_probabilities['Positive'] * 100
    negative_prob = sentiment_probabilities['Negative'] * 100

    positive_probs.append(positive_prob)
    negative_probs.append(negative_prob)

    print("Sentence:", sentence)
    print("Positive Probability:", int(positive_prob), "%")
    print("Negative Probability:", int(negative_prob), "%")

# Plotting
x = ['Positive', 'Negative']

plt.bar(x, [positive_probs[0], negative_probs[0]], color=['green', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Probability (%)')
plt.title('Sentiment Analysis Result')
plt.tight_layout()
plt.show()
```

### 키워드 및 핵심 문구 추출
``` c 
# Load the model
kw_model = KeyBERT('all-mpnet-base-v2')

# Use KeyphraseCountVectorize
from tqdm import tqdm

def apply_keybert(sentence):
    keywords = kw_model.extract_keywords(sentence, vectorizer=KeyphraseCountVectorizer(), stop_words='english', top_n=3)
    return ', '.join([keyword for keyword, score in keywords]) # 키워드는 중요도 내림차순으로 최대 3개까지 저장된다.

# Creating Columns and Storing Keywords
df['keywords'] = df['Review'].apply(apply_keybert)

# Counting Keywords and Keyphrase

from collections import Counter

# 모든 인덱스의 'keywords' 컬럼을 합쳐서 단어들을 카운트하는 함수
def count_all_keywords(dataframe):
    all_keywords = dataframe['keywords'].str.split(', ').sum()
    keyword_counts = Counter(all_keywords)
    sorted_keyword_counts = keyword_counts.most_common() # 키워드 빈도가 많은 순으로 내림차순으로 정렬한다.
    return sorted_keyword_counts

# 모든 인덱스의 'keywords' 컬럼에 있는 단어들을 카운트
sorted_all_keyword_counts = count_all_keywords(beer_Wired_iStout_pre)
```
---
## Reference
#### [1. Hugging_Face_T5_Guide](https://huggingface.co/docs/transformers/model_doc/t5)
#### [2. T5_Paper](https://arxiv.org/pdf/1910.10683v3.pdf)
#### [3. SimpleT5 github](https://github.com/Shivanandroy/simpleT5/tree/main)
#### [4. Data_Labeling_VADER](https://medium.com/analytics-vidhya/sentiment-analysis-with-vader-label-the-unlabeled-data-8dd785225166)
#### [5. Data_Labeling_GPT](https://towardsdatascience.com/can-chatgpt-compete-with-domain-specific-sentiment-analysis-machine-learning-models-cdcd9937b460)
#### [6. Data_Labeling_Alpaca](https://www.youtube.com/watch?v=JzBR8oieyy8&t=117s)
#### [7. The Most Common Evaluation Metrics In NLP](https://medium.com/towards-data-science/the-most-common-evaluation-metrics-in-nlp-ced6a763ac8b)
#### [8. Pytorch Multi GPU](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)
#### [9. Data Augmentation 기법](https://maelfabien.github.io/machinelearning/NLP_8/#when-should-we-use-data-augmentation)
#### [10. Gradio 시연](https://levelup.gitconnected.com/sharing-your-machine-learning-or-deep-learning-projects-with-users-with-gradio-10b42588a55d)
---
## 프로젝트 결과물 모음

|No|내용|깃허브|
|-|-|-|
|1|데이터 수집|[📂](https://github.com/sgr1118/Bert_beer_sentiment_anlysis/tree/main/Data)|
|2|데이터 라벨링|[📂](https://github.com/sgr1118/Bert_beer_sentiment_anlysis/tree/main/Data/Data_labeling_test)|
|3|파인 튜닝|[📂](https://github.com/sgr1118/Bert_beer_sentiment_anlysis/tree/main/Models/Step1_Bert_train)|
---
## 프로젝트 개선 요구 사항

### 1. 중립 라벨링 추가
- 좀더 세분화된 감정 분류를 수행하기위해 중립 라벨링 기준을 확립하고 적용할 예정

### 2. 핵심 문구 추출 속도 증가
- KeyBERT를 사용하여 핵심 문구 추출 시간이 데이터가 많아질수록 길어진다. 실시간 응답으로 키워드 추출 결과를 보여주기 힘들다는 단점이있다.

## 프로젝트 후원 : (주)모두의연구소, K-디지털 플랫폼
---
본 프로젝트는 모두의연구소와 K-디지털 플랫폼으로부터 지원받았습니다.
