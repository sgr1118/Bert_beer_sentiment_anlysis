from collections import defaultdict
import spacy
import ast
import pandas as pd


# 키워드 추출된 데이터프레임 불러오기
df = pd.read_csv('../../Data/Preprocessed_data/pp_selected_reviews_JJ_NN.csv')


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


# 명사 위주의 동의어 집계 적용
filtered_and_merged_keywords, merged_keyword_mappings = merge_similar_keywords_noun_weight(df)


# 어떤 키워드가 병합됐는지 매핑하여 시각적 확인
print("▶ 병합된 키워드 그룹 개수:", len(filtered_and_merged_keywords))
print("▶ 병합된 키워드 매핑:")
for key, value in filtered_and_merged_keywords.items():
    print(f"병합된 키워드: {key} [{value}]")
    print(merged_keyword_mappings[key])
    print()