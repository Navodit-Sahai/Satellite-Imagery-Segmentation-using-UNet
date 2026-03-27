# Satellite Imagery Segmentation
 
Semantic segmentation of aerial/satellite imagery using a U-Net model in TensorFlow/Keras. The model classifies each pixel into one of 6 land-cover classes.
 
---
 
## Classes
 
| Class | Color |
|---|---|
| Water | `#E2A929` |
| Land | `#8429F6` |
| Road | `#6EC1E4` |
| Building | `#3C1098` |
| Vegetation | `#FEDD3A` |
| Unlabeled | `#9B9B9B` |
 
---
 
## Dataset
 
[Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) from Kaggle.
 
- 7 tiles, each with RGB images and corresponding RGB masks
- Images are patched into **256×256** patches before training
 
---
 
## Project Structure
 
```
├── Satellite_Imagery_Segmentation.ipynb   # Main notebook
├── my_model.h5                            # Saved trained model
└── README.md
```
 
---
 
## Setup
 
```bash
pip install tensorflow segmentation-models patchify opendatasets scikit-learn
```
 
---
 
## Pipeline
 
**1. Data Preprocessing**
- Load images and masks from each tile
- Patch into 256×256 crops using `patchify`
- Normalize images to [0, 1]
- Convert RGB masks → integer class labels → one-hot encoded
 
**2. Model — U-Net**
- Custom U-Net with 5 encoder/decoder levels
- Filter sizes: 8 → 16 → 32 → 64 → 128
- Dropout regularization at each level
- Softmax output over 6 classes
 
**3. Training**
- Loss: `DiceLoss + CategoricalFocalLoss` (from `segmentation_models`)
- Metric: Jaccard Coefficient (IoU)
- Optimizer: Adam
- Epochs: 10, Batch size: 16
 
**4. Evaluation**
- Side-by-side visualization of input image, ground truth mask, and predicted mask
 
---
 
## Custom Metric
 
```python
def jaccard_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(tf.reshape(y_true, [-1]) * tf.reshape(y_pred, [-1]))
    union = tf.reduce_sum(tf.reshape(y_true, [-1])) + tf.reduce_sum(tf.reshape(y_pred, [-1])) - intersection
    return (intersection + 1e-7) / (union + 1e-7)
```
 
---
 
## Loading the Model for Inference
 
```python
from tensorflow.keras.models import load_model
 
model = load_model("my_model.h5",
                   custom_objects={"jeccard_coef": jaccard_coef},
                   compile=False)
 
predictions = model.predict(x_test)
predicted_masks = np.argmax(predictions, axis=-1)
```
 
---
 
## Results
 
Predictions are visualized as a 3-column grid — original image, ground truth, and predicted mask — for qualitative evaluation.
 
