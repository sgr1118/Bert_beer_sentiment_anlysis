## 데이터 전처리를 수행하는 Python 파일입니다.

### 수행하는 전처리 목록
- "ain't": "is not", "aren't": "are not" 같은 문자에 대하여 텍스트 정규화
- 대문자 > 소문자
- 맥주의 용량 제거
- 특정 문구 제거(---rated via beer buddy for iphone)
- 괄호로 닫힌 문자열 제거
- 지정한 문자 제외 공백으로 전환
- 불필요한 공백 제거
- nltk stopwords 삭제
