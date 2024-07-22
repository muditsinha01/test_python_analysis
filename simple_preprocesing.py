import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_inspect_data(file_path):
    """
    Load and inspect the initial dataset.
    """
    dataset = pd.read_csv(file_path)
    print(dataset.info())
    print(dataset.head())
    return dataset

def clean_data(dataset):
    """
    Perform data cleaning, including removing duplicates and handling missing values.
    """
    dataset_cleaned = dataset.dropna().drop_duplicates()
    return dataset_cleaned

def preprocess_data(dataset):
    """
    Tokenize source code and apply one-hot encoding to the vulnerability type.
    """
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(dataset['Source code'])
    dataset['Code_Tokens'] = tokenizer.texts_to_sequences(dataset['Source code'])

    max_length = max([len(seq) for seq in dataset['Code_Tokens']])
    dataset['Code_Tokens_Padded'] = pad_sequences(dataset['Code_Tokens'], maxlen=max_length, padding='post').tolist()

    ohe = OneHotEncoder(sparse=False)
    dataset['Vulnerability_OHE'] = list(ohe.fit_transform(dataset[['Vulnerability type']]))

    return dataset, tokenizer

def build_model(tokenizer, max_length, output_dim):
    """
    Build and compile the LSTM model.
    """
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model

def train_and_evaluate_model(model, dataset):
    """
    Train the model and evaluate its performance.
    """
    X = np.array(dataset['Code_Tokens_Padded'].tolist())
    y = np.array(dataset['Vulnerability_OHE'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')

def main():
    """
    Main function to execute the steps.
    """
    file_path = 'FormAI_dataset.csv'
    dataset = load_and_inspect_data(file_path)
    dataset_cleaned = clean_data(dataset)
    dataset_preprocessed, tokenizer = preprocess_data(dataset_cleaned)
    model = build_model(tokenizer, max_length=max([len(seq) for seq in dataset_preprocessed['Code_Tokens']]), output_dim=len(dataset_preprocessed['Vulnerability_OHE'][0]))
    train_and_evaluate_model(model, dataset_preprocessed)

if __name__ == "__main__":
    main()
