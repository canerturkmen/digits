### Digits Recognizer Classification Example

Hi! Here's a very short example for how the SciPy stack and scikit-learn can be used for a solution to the Digits Recognizer Challenge on [Kaggle](www.kaggle.com).

Any questions, [email me](mailto:turkmen.ac@gmail.com)

#### Methods

The example works by:
- First using a non-negative matrix factorization with 10 components for feature extraction. Due to limitations with computing resources, NMF is trained only with 1K rows of the data.
- The example then features both a SVM classifier, and a Random Forest classifier. The first classifier yields an accuracy of 84%, whereas the second: 88%.
- There is also a neat function for drawing the digit in matplotlib's imshow.

Enjoy 
