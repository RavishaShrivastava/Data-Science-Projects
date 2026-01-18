# Ancient Language Detection using Machine Learning

## Project Overview

This project focuses on **automatic detection of ancient languages** from text data using Machine Learning techniques.
The model is trained to classify text into one of several ancient languages based on learned linguistic patterns.

The goal of this project is to explore **text classification**, **language identification**, and **dataset structuring** for multi-class problems involving historical and ancient scripts.

---

## Languages Covered

The model currently supports the following ancient languages:

* **Arabic**
* **Egyptian**
* **Sanskrit**
* **Latin**

---

## Dataset Structure

The dataset is organized in a structured and scalable format to support training, validation, and testing.

```
ancient_lang/
│
├── Ancient/
│   ├── train/
│   │   ├── arabic/
│   │   ├── egyptian/
│   │   ├── sanskrit/
│   │   └── latin/
│   │
│   ├── valid/
│   │   ├── arabic/
│   │   ├── egyptian/
│   │   ├── sanskrit/
│   │   └── latin/
│   │
│   └── test/
│       ├── arabic/
│       ├── egyptian/
│       ├── sanskrit/
│       └── latin/
```

Each folder contains text samples corresponding to its respective language.

## Model & Approach

* **Problem Type:** Multi-class text classification
* **Input:** Text samples from ancient languages
* **Output:** Predicted language label
* **Approach:**

  * Text preprocessing and normalization
  * Feature extraction from textual data
  * Supervised learning for classification

## Results

* **Overall Accuracy:** **75%**
* The model demonstrates a strong ability to distinguish between multiple ancient languages despite similarities in structure and vocabulary.

## Future Improvements

* Expand the dataset with more samples per language
* Add support for additional ancient languages
* Experiment with deep learning models (LSTM, Transformer-based models)
* Improve preprocessing for script-specific nuances

## Technologies Used

* Python
* Machine Learning libraries (e.g., Scikit-learn / TensorFlow / PyTorch)
* NLP techniques for text processing

## Conclusion

This project demonstrates a practical implementation of **language detection for ancient texts**, showcasing the potential of Machine Learning in historical and linguistic research domains.

---

## 📬 Contact

Feel free to connect or contribute to this project through GitHub.

⭐ If you find this project interesting, consider giving it a star!
