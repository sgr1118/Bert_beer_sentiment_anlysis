import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, BertTokenizer  # 수정
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import AdamW
import time

# colab에서 git Clone을하여 사용하시는걸 추천 드립니다.
# 먼저 Data 경로에 있는 preprocessing.py를 실행시켜주세요!

# 데이터셋 만들기

df = pd.read_csv('/mnt/c/Users/GrSon/Desktop/sentiment_analysis/Bert_beer_sentiment_anlysis/Data/Preprocessed_data/preprocess.csv') # load to preprocess.csv

def load_review_dataset(random_seed = 1,):
# def load_review_dataset(random_seed = 1,):
    # df = df

    # train, test 분류
    X_train, X_test, y_train, y_test = \
        train_test_split(df['Review'].tolist(), df['MultinomialNB_label'].tolist(),
                         shuffle=True, test_size=0.2, random_state=random_seed, stratify=df['MultinomialNB_label'])

    # transform to pandas dataframe
    train_data = pd.DataFrame({'source_text': X_train, 'label': y_train})
    test_data = pd.DataFrame({'source_text': X_test, 'label': y_test})

    # return train_data, val_data, test_data
    return train_data, test_data

# create data
train_df, test_df = load_review_dataset(1)
print(train_df.shape, test_df.shape)

# Fine_Tunning Start!
# Record start time
start_time = time.time()

# 데이터 불러오기
random_seed = 1
reviews = train_df['source_text'].tolist()
labels = train_df['label']. map({ 'Positive' : 1 , 'Negative' : 0 }).tolist()

# 데이터 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(reviews, labels, \
                         shuffle=True, test_size=0.2, random_state=random_seed, stratify = train_df['label'])

# 토크나이저 초기화
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

 # Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# Create torch dataset
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataloaders
train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = True, # Whether the model returns attentions weights.
    output_hidden_states = True, # Whether the model returns all hidden-states.
)
model = model.to('cuda')  # if GPU is available

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Lists to store training and validation loss values
train_losses = []
val_losses = []

# Training loop
for epoch in range(3):  # number of epochs
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['label'].to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Save the model
model.save_pretrained('sentiment_model_BERT')

# Save tokenizer configuration and vocabulary
tokenizer.save_pretrained('sentiment_model_BERT')  # 추가

# Record end time
end_time = time.time()

print("Time required to fine-tune: ", end_time - start_time)