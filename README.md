# Russian News Summarization with BERT Encoder and Transformer Decoder

Проект посвящен абстрактивной суммаризации русскоязычных новостей из датасета `IlyaGusev/gazeta`. Основная цель эксперимента — реализовать собственную encoder-decoder архитектуру на базе предобученного BERT-энкодера и кастомного Transformer-декодера, затем честно сравнить ее с простыми и предобученными baseline-моделями.

Ноутбук: `Transformer.ipynb`

## Задача

Для каждой новости модель должна сгенерировать короткое summary, близкое к эталонному описанию из датасета. Проект сфокусирован не только на итоговом качестве генерации, но и на полном исследовательском пайплайне: подготовка данных, обучение, сохранение лучшей модели, декодирование, метрики и анализ ошибок.

## Данные

Используется датасет `IlyaGusev/gazeta` с Hugging Face Datasets.

Текущая конфигурация:

```python
dataset_fraction = "train"
batch_size = 64
final_num_epochs = 8
max_eval_samples = 100
```

Данные делятся на независимые части:

- `train` — обучение модели;
- `validation` — выбор лучшего чекпоинта;
- `test` — финальная оценка качества.

## Архитектура

Кастомная модель состоит из:

- предобученного BERT-энкодера `deepvk/bert-base-uncased`;
- собственного `nn.TransformerDecoder`;
- token embeddings и positional embeddings для decoder-части;
- линейной головы в словарь токенизатора;
- `LogSoftmax` и `NLLLoss` для обучения.

В генерации реализованы:

- greedy decoding;
- top-k sampling;
- top-p sampling;
- beam search;
- `repetition_penalty`;
- `no_repeat_ngram_size`;
- `min_len`.

## Обучение

Финальный запуск использует:

- mixed precision training через AMP;
- staged fine-tuning: первые эпохи BERT заморожен, затем размораживается;
- TensorBoard logging;
- сохранение лучшего чекпоинта по validation loss.

Последний сохраненный прогон:

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 6.8784 | 6.5916 |
| 2 | 6.3571 | 6.1436 |
| 3 | 5.9552 | 5.7972 |
| 4 | 5.6118 | 5.5209 |
| 5 | 5.3190 | 5.2721 |
| 6 | 5.0640 | 5.0779 |
| 7 | 4.8352 | 4.8987 |
| 8 | 4.6221 | 4.7426 |

Validation loss стабильно снижается, что показывает обучаемость кастомной архитектуры.

## Метрики

Финальная оценка проводится на test split. Сравниваются три подхода:

- `Lead-2 baseline` — первые два предложения новости;
- `Custom BERT + Transformer Decoder` — кастомная модель проекта;
- `ruT5-small` — предобученная seq2seq baseline-модель.

Результаты последнего сохраненного запуска:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | SacreBLEU | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| Lead-2 baseline | 0.2171 | 0.0725 | 0.2133 | 8.1009 | 0.7088 |
| Custom BERT + Transformer Decoder | 0.0662 | 0.0067 | 0.0675 | 1.2166 | 0.6620 |
| ruT5-small | 0.0640 | 0.0022 | 0.0641 | 2.1855 | 0.6674 |

## Выводы

Кастомная модель обучается и генерирует связные summary, но пока заметно уступает `Lead-2 baseline`. Это ожидаемо для новостной суммаризации: первые предложения часто уже содержат главную информацию, поэтому extractive baseline оказывается очень сильным.

При этом кастомная модель сравнима с `ruT5-small` на выбранной test-подвыборке по ROUGE-L, но качественный анализ показывает проблему factual consistency: модель иногда сохраняет тему новости, но путает конкретные имена, клубы, числа или участников события.

Главный результат проекта — не production-ready summarizer, а полный NLP-эксперимент с собственной Transformer-архитектурой, честными baseline-сравнениями и анализом ограничений.

## Ограничения

- BERT не является seq2seq-моделью, поэтому decoder обучается практически с нуля.
- Новости из `gazeta` часто хорошо суммаризируются первыми предложениями, из-за чего `Lead-2` является сильным baseline.
- Модель склонна к factual errors: может путать имена, команды, страны и численные факты.
- ROUGE не полностью отражает фактическую корректность summary.

## Возможные улучшения

- Увеличить число эпох или использовать learning rate schedule.
- Попробовать `batch_size=32` для большего числа optimizer updates на эпоху.
- Добавить label smoothing.
- Использовать pretrained seq2seq checkpoint как основной baseline: `ruT5`, `mT5`, `MBART`.
- Добавить constrained decoding или reranking для уменьшения factual hallucinations.
- Расширить test evaluation с 100 примеров до полного test split.
- Вынести обучение и оценку из ноутбука в отдельные Python-модули.

## Запуск

Рекомендуемая среда:

- Google Colab Pro;
- GPU: A100 80GB или H100;
- Python 3;
- PyTorch + CUDA.

Основные зависимости устанавливаются в ноутбуке:

```python
!pip install transformers datasets evaluate
!pip install rouge_score sacrebleu bert_score
```

После смены GPU или изменения архитектуры рекомендуется перезапустить runtime и выполнить ноутбук с начала.

## Portfolio Summary

Built and evaluated a custom BERT-based encoder-decoder model for Russian news summarization. Implemented a Transformer decoder with positional embeddings, multiple decoding strategies, mixed precision training, checkpointing, and evaluation with ROUGE, SacreBLEU, and BERTScore. Compared the custom model against `ruT5-small` and a strong `Lead-2` baseline, then analyzed repetition and factual consistency issues.
