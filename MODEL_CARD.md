# Model card: page-flip ConvNet

Sections follow *Model Cards for Model Reporting* (Mitchell et al., 2019).

## Model details

- **Developer:** Siyuan David Zhu. The model was built in 2022 as an applied-ML
  residency project and packaged as a library in 2026.
- **Version:** 0.1.0.
- **Type:** a convolutional neural network for binary image classification.
  - Four convolution layers (11 × 11, 5 × 5, 1 × 1, 1 × 1) with batch
    normalisation and max pooling.
  - Two dense layers of 1,024 units with 20% dropout, and a sigmoid output.
  - 11.9 million parameters.
- **Training:** Adam (learning rate 0.01), binary cross-entropy, 25 epochs,
  batch size 128.
- **Implementations:** Keras 3 (runs on TensorFlow, JAX or PyTorch) and a
  native PyTorch version. Given the same weights, they agree to within
  0.00001.
- **License:** MIT.

## Intended use

- **Primary use:** deciding, in a book-digitization app, whether a page is
  being flipped, so that the full-resolution capture happens after the flip.
- **Out of scope:**
  - reading or interpreting document content;
  - decisions about people;
  - capture setups other than the one the model was trained on.

## Factors

The capture setup is the main factor: the device, lighting, paper and page
size. All training and test images share one setup, so the model's
performance under other setups is unknown.

## Metrics

- Accuracy, plus per-class precision, recall and F1, at a 0.5 threshold.
- 5-fold cross-validation accuracy (mean and standard deviation) on the
  training set.
- The test set is close to balanced (49% `flip`), so accuracy is not
  dominated by one class.

## Evaluation data

597 held-out images (290 `flip`, 307 `notflip`), supplied as a separate test
folder with the residency project's image set. They are not shipped with this
repo.

## Training data

2,392 labelled images from the same image set and capture setup. They are not
shipped with this repo.

## Quantitative analyses (2022)

| | Precision | Recall | F1 |
|---|---|---|---|
| `flip` | 0.99 | 0.98 | 0.98 |
| `notflip` | 0.98 | 0.99 | 0.98 |

- **Test accuracy:** 0.98.
- **5-fold cross-validation accuracy:** 96.2% (std 3.3); the folds scored
  96.9%, 98.1%, 99.8%, 96.4% and 90.0%.

## Ethical considerations

- The model sees images of book pages and predicts only whether a page is
  being flipped. It makes no predictions about people.
- An error costs a blurred capture or a short delay.

## Caveats and recommendations

- The cross-validation score may be optimistic, because images taken moments
  apart can land in both training and validation folds. Treat the held-out
  test score as the reference figure.
- Each image is judged on its own. Before using the model with a new device
  or setup, check its accuracy on images from that setup.
- To retrain, run
  `doc-motion train --data <folder with train/ and test/> --out runs/`. Each
  run records its config, fold scores, training loss and test report in
  `runs/<date>_<time>_<backend>/metrics.json`.
