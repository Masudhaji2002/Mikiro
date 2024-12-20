import numpy as np
from transformers import BertTokenizer, BertForTokenClassification
from terms_extractor.dl_extractor.heuristic_validator import HeuristicValidator
import torch
import os

# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к модели и словарю
MODEL_PATH = "terminator/models/dl_extractor"
VOCAB_PATH = os.path.join(MODEL_PATH, "vocab.txt")

# Проверка наличия файлов
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Файл vocab.txt не найден по пути: {VOCAB_PATH}")

# Загрузка токенизатора и модели
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", do_lower_case=False)
model = BertForTokenClassification.from_pretrained(MODEL_PATH).to(device)

# Исправление BIO-разметки для подслов
def fix_subword_labels(tokens, predicted_labels):
    fixed_labels = []
    prev_label = "O"
    for token, label in zip(tokens, predicted_labels):
        if token.startswith("##"):
            if prev_label == "B-TERM":
                fixed_labels.append("I-TERM")
            else:
                fixed_labels.append(prev_label)
        else:
            fixed_labels.append(label)
            prev_label = label
    return fixed_labels

# Интеграция с эвристическим валидатором
def apply_heuristics(results):
    validator = HeuristicValidator()
    return validator.validate(results)

# Предсказание и обработка
def predict_and_display(text, model, tokenizer, max_length=128):
    # Токенизация текста
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_length - 2]
    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_ids)

    # Паддинг
    padding_length = max_length - len(input_ids)
    input_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    # Подготовка входов
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    attention_mask_tensor = torch.tensor([attention_mask]).to(device)

    # Предсказание
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits = outputs.logits

    # Постобработка предсказаний
    predicted_labels = torch.argmax(logits, dim=2).squeeze().tolist()
    label2id = {0: "O", 1: "B-TERM", 2: "I-TERM"}
    decoded_labels = [label2id[label] for label in predicted_labels]

    # Исправление разметки подслов
    fixed_labels = fix_subword_labels(input_tokens, decoded_labels)

    # Подготовка результатов для валидации
    results = [(token, label) for token, label in zip(input_tokens, fixed_labels) 
               if token not in ["[CLS]", "[SEP]", "[PAD]"]]

    # Применение эвристик
    validated_results = apply_heuristics(results)

    # Вывод окончательных результатов
    print("Результаты после применения эвристик:")
    for token, label in validated_results:
        print(f"{token}: {label}")

# Тестовый пример
if __name__ == "__main__":
    sample_text = "Дом из кирпича"
    predict_and_display(sample_text, model, tokenizer)
