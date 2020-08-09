# Audio classifier

Initially it was a solution for [Speech Activity Classifier competition](https://www.kaggle.com/c/silero-audio-classifier) on Kaggle

Here it is presented in a module-style structure.

## Requirements
* scipy 1.4.1+
* numpy 1.17.4+
* pandas 0.25.3+
* tqdm 4.40.0+
* librosa 0.8.0+
* pytorch 1.5.0+

## Usage

```python
from AudioClassifier import AudioClassifier
clf = AudioClassifier()
clf.train(train_data_df, train_data_folder)
clf.predict(test_data_df, test_data_folder)
```