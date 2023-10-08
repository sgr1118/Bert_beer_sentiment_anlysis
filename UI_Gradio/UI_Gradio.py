import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
import gradio as gra
import gradio as gr
from itertools import islice
from collections import defaultdict
import spacy
import ast

# 모델 로드
model = BertForSequenceClassification.from_pretrained('/content/drive/MyDrive/sentiment_bert/sentiment_model_BERT_Attention_v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 토크나이저 초기화
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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

def sentiment_analysis(text):
    sentiment_probabilities = analyze_sentiment(text)
    return sentiment_probabilities

################################################################################################################

nlp = spacy.load('en_core_web_lg')

# 키워드의 초기 빈도 수를 계산
def get_initial_counts(keywords):
    initial_counts = defaultdict(int)
    for keyword in keywords:
        initial_counts[keyword] += 1
    return initial_counts

# 명사 토큰 저장하고 각 유사성을 계산하여 명사 유사성 임계값을 초과하는지 확인
# 형용사의 차이는 개인차에 의해 발생하고 명사는 특징 자체를 나타내기에 집중 그룹핑 대상이 됨
def custom_similarity(doc1, doc2, noun_similarity_threshold):
    doc1_nouns = [token for token in doc1 if token.pos_ == 'NOUN']
    doc2_nouns = [token for token in doc2 if token.pos_ == 'NOUN']
    noun_similar = any(t1.similarity(t2) > noun_similarity_threshold for t1 in doc1_nouns for t2 in doc2_nouns)
    return noun_similar

# 유사한 키워드들을 병합하고 키워드 빈도 수와 매핑 저장
def merge_keywords(keyword_counts, noun_similarity_threshold):
    merged_keywords = defaultdict(int)
    keyword_mappings = defaultdict(list)
    keyword_docs = {}

    # 이미 등록된 키워드는 nlp 계산(임베딩)하지 않고 바로 리턴 (Memoization)
    def get_keyword_doc(keyword):
        if keyword not in keyword_docs:
            keyword_docs[keyword] = nlp(keyword)
        return keyword_docs[keyword]


    # 초기 빈도 딕셔너리를 순회하며 각 키워드에 대해 임베딩
    for keyword, count in keyword_counts.items():
        doc1 = get_keyword_doc(keyword)
        merged_or_found_similar = False

        # 이미 병합된 키워드 목록에서 해당 키워드와 비교
        for merged_keyword in list(merged_keywords):
            # 이미 병합된 키워드에 현재 키워드가 포함되어 있다면 빈도를 더함
            if keyword == merged_keyword:
                merged_keywords[merged_keyword] += count
                merged_or_found_similar = True
                keyword_mappings[merged_keyword].append(keyword)
                break

            doc2 = get_keyword_doc(merged_keyword)

            # 유사도가 임계값보다 큰 경우, 두 키워드를 같은 그룹으로 묶음
            if custom_similarity(doc1, doc2, noun_similarity_threshold):
                merged_keywords[merged_keyword] += count
                merged_or_found_similar = True
                keyword_mappings[merged_keyword].append(keyword)
                break

        # 현재 키워드가 기존 병합된 키워드 목록에 포함되지 않고, 유사한 키워드도 없다면 새로운 항목으로 추가
        if not merged_or_found_similar:
            merged_keywords[keyword] = count
            keyword_mappings[keyword] = [keyword]

    return merged_keywords, keyword_mappings

def merge_similar_keywords_noun_weight(dataframe, noun_similarity_threshold=0.7):
    # 문자열로 표현된 리스트를 실제 리스트로 변환
    dataframe['Keywords_List'] = dataframe['Keywords'].apply(lambda x: ast.literal_eval(x))
    all_keywords = dataframe['Keywords_List'].sum()

    # 빈도 수 계산 및 병합
    initial_counts = get_initial_counts(all_keywords)
    filtered_and_merged_keywords, merged_keyword_mappings = merge_keywords(initial_counts, noun_similarity_threshold)
    sorted_merged_keywords = sorted(filtered_and_merged_keywords.items(), key=lambda x: x[1], reverse=True)

    # 변환된 'Keywords_List' 컬럼 삭제
    dataframe.drop('Keywords_List', axis=1, inplace=True)

    return dict(sorted_merged_keywords), dict(merged_keyword_mappings)

#########################################################################################################################

from itertools import islice

beer_df = pd.read_csv('../../Data/Preprocessed/pp_selected_reviews_JJ_NN.csv')

def show_keywords(beer_name, sentiment, flag=0):
    one_beer_df = beer_df[beer_df['Beer_name'] == beer_name]
    df = one_beer_df[one_beer_df['MultinomialNB_label'] == sentiment]
    df.reset_index(drop=True, inplace=True)
    keywords, mappings = merge_similar_keywords_noun_weight(df)

    if flag == 1:
        return "\n".join([f"{k}: {v}" for k, v in islice(keywords.items(), None, 10)])
    else:
        return "\n".join([f"{k}: {v}" for k, v in islice(keywords.items(), None, 10)]), mappings

def keyword_mappings(beer_name, sentiment, keyword):
    _, mappings = show_keywords(beer_name, sentiment)
    mapped_keywords = mappings.get(keyword, "해당 키워드에 매핑된 문자열이 없습니다.")
    return mapped_keywords

beer_names = list(beer_df['Beer_name'].unique())
sentiments = ["Positive", "Negative"]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            beer_name_dropdown = gr.Dropdown(choices=beer_names, label="Beer_name", info="Choose the beer you want to see the summary keyword.")
            sentiment_dropdown = gr.Dropdown(choices=sentiments, label="Sentiment", info="Choose Positive or Negative sentiment.")
            keyword_input = gr.Textbox(label="Keyword", info="Enter a keyword to see its mappings.")
        with gr.Column():
            output_keywords = gr.Textbox(label="Summary keywords (Top 10)")
            output_mapping = gr.Textbox(label="Keyword mapping")
    with gr.Row():
        with gr.Column():
            review_button = gr.Button("Review Topic")
        with gr.Column():
            mapping_button = gr.Button("Keyword mapping")

    review_button.click(fn=lambda beer_name, sentiment: show_keywords(beer_name, sentiment, flag=1),
                    inputs=[beer_name_dropdown, sentiment_dropdown],
                    outputs=output_keywords)
    mapping_button.click(keyword_mappings,
                        inputs=[beer_name_dropdown, sentiment_dropdown, keyword_input],
                        outputs=output_mapping)

# Tab 1
app1 = gra.Interface(fn=sentiment_analysis,
                     inputs="text",
                     outputs="label",
                     live=True,
                     title = "Beer Sentiment Analysis")

# Tab 2
app2 = demo

# 탭 1과 2를 그룹화
tabbed = gra.TabbedInterface([app1, app2],
                             [ 'Sentiment Analysis' , 'Keyword Extraction'])
tabbed.launch(share=True)