# MELD Data Preparation

This project uses the **MELD (Multimodal EmotionLines Dataset)** for multimodal
emotion recognition.

For the development and training of the current prototype, only the MELD
**training split** was used. The training data was later divided by dialogue
into training and validation subsets in order to prevent utterances from the
same dialogue from appearing in both sets.

The model uses two modalities:

- Text
- Audio

Video files from MELD are used as the source of the audio signal, but visual
features are not used by the current model.

---

## Dataset Structure

The processed dataframe contains the following fields:

- `Sr No.`
- `Utterance`
- `Speaker`
- `Emotion`
- `Sentiment`
- `Dialogue_ID`
- `Utterance_ID`
- `Season`
- `Episode`
- `StartTime`
- `EndTime`
- `sound_location`

After preprocessing, no missing values were detected in any of these columns.

---

## Emotion Distribution

The processed training dataset contains **9,988 utterances** distributed among
the seven MELD emotion classes.

| Emotion | Number of Utterances |
|---|---:|
| Neutral | 4,709 |
| Joy | 1,743 |
| Surprise | 1,205 |
| Anger | 1,109 |
| Sadness | 683 |
| Disgust | 271 |
| Fear | 268 |
| **Total** | **9,988** |

The class distribution is clearly imbalanced. `neutral` is the dominant class,
while `fear` and `disgust` contain substantially fewer examples.

<img width="747" height="537" alt="image" src="https://github.com/user-attachments/assets/39fb42f5-a3a5-4aa6-8408-0c0ec4983325" />

This imbalance is important when evaluating the emotion classifier. Therefore,
in addition to accuracy, Macro F1 was used as an important model-selection
metric because it gives equal importance to each emotion class.

---
## Text Encoding Normalization

During data preprocessing, some utterances contained incorrectly represented
punctuation and special characters. In particular, characters associated with
Windows-1252 encoding, such as curly apostrophes and quotation marks, required
normalization before text feature extraction.

A normalization function was applied to each text entry. The preprocessing
consisted of three main steps:

1. **Character encoding correction:** Text was first encoded using `latin1`
   and decoded using `cp1252` to correct Windows-1252-style characters.

2. **Punctuation normalization:** Common typographic characters were replaced
   with their ASCII equivalents. This included curly apostrophes, quotation
   marks, and en/em dashes.

3. **Unicode normalization:** Unicode NFKD normalization was applied, followed
   by ASCII encoding. This converted accented characters when possible and
   removed remaining characters that could not be represented in ASCII.

The normalization was implemented as:

```python
import unicodedata

def to_ascii(text):
    if not isinstance(text, str):
        return text

    # Fix Windows-1252 style characters
    text = text.encode(
        "latin1",
        errors="ignore"
    ).decode(
        "cp1252",
        errors="ignore"
    )

    # Replace common punctuation with ASCII equivalents
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Normalize remaining Unicode characters to ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode(
        "ascii",
        errors="ignore"
    ).decode("ascii")

    return text
```
---

## Dataset Split Used in This Project

The original MELD dataset provides separate training, validation, and testing
data. During development of this project, however, problems were encountered
when attempting to decode the media files from the downloaded MELD test split.
The corresponding audio/video files could not be opened reliably, possibly
because of a media formatting, extraction, or file integrity issue.

Rather than modifying the model or preprocessing procedure specifically to
accommodate these files, the current prototype uses the successfully decoded
MELD training data.

The available training dialogues were divided into two subsets:

- **80% of the dialogues** for model training
- **20% of the dialogues** for validation/evaluation

The split was performed at the dialogue level rather than at the individual
utterance level:

```python
from sklearn.model_selection import train_test_split

dialogue_ids = df["Dialogue_ID"].unique()

train_ids, val_ids = train_test_split(
    dialogue_ids,
    test_size=0.2,
    random_state=42
)

```

## Dataset Source and Citation

This project uses the **MELD (Multimodal EmotionLines Dataset)**, a multimodal
dataset designed for emotion recognition in multiparty conversations.

MELD extends the original EmotionLines dataset by incorporating audio and
visual information in addition to textual dialogue. The complete dataset
contains more than 1,400 dialogues and 13,000 utterances obtained from the
television series *Friends*. Each utterance is annotated with one of seven
emotion categories:

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

The dataset also provides sentiment annotations for each utterance.

Official MELD project page:
https://affective-meld.github.io/

The MELD authors request that the following two works be cited when using the
dataset:

### References

1. Poria, S., Hazarika, D., Majumder, N., Naik, G., Mihalcea, R., & Cambria, E.
   (2018). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
   Conversation.*

2. Chen, S. Y., Hsu, C. C., Kuo, C. C., & Ku, L. W. (2018).
   *EmotionLines: An Emotion Corpus of Multi-Party Conversations.*
   arXiv preprint arXiv:1802.08379.
