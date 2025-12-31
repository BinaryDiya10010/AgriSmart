import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

TRAIN_DIR = 'datasets/Train/Train'
TEST_DIR = 'datasets/Test/Test'
VAL_DIR = 'datasets/Validation/Validation'
MODEL_DIR = 'models/disease_detection'
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_disease.h5')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.json')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001


def count_images():
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    total_images = 0
    for split_name, split_dir in [('Training', TRAIN_DIR), ('Test', TEST_DIR), ('Validation', VAL_DIR)]:
        if not os.path.exists(split_dir):
            print(f"\nWARNING: {split_name} directory not found: {split_dir}")
            continue
        
        print(f"\n{split_name} Set:")
        split_total = 0
        
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) if f.endswith('.jpg')])
                print(f"  {cls:15s}: {count:4d} images")
                split_total += count
        
        print(f"  {'Total':15s}: {split_total:4d} images")
        total_images += split_total
    
    print(f"\n{'GRAND TOTAL':17s}: {total_images:4d} images")
    print("="*80 + "\n")
    
    return total_images


def create_model(num_classes=3):
    print("\nBuilding model architecture...")
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(0.5, name='dropout')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


def prepare_data_generators():    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def train_model():
    print("\n" + "="*80)
    print("TRAINING DISEASE DETECTION MODEL")
    print("="*80)
    
    total_images = count_images()
    
    train_gen, val_gen, test_gen = prepare_data_generators()
    num_classes = len(train_gen.class_indices)
    
    model = create_model(num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"Loss: Categorical Crossentropy")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    '''print(f"\nCallbacks:")
    print(f"ModelCheckpoint (save best model)")
    print(f"EarlyStopping (patience=10)")
    print(f"ReduceLROnPlateau (patience=5)")
    
    print("\n" + "="*80)
    print("Training Sgtarted")
    print("="*80)
    '''
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    '''print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    '''
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"Class indices saved: {CLASS_INDICES_PATH}")
    
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy'])
    }
    
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved: {HISTORY_PATH}")
    print(f"Model saved: {MODEL_PATH}")
    '''
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)
    print(f"\nResults:")
    print(f"   - Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"   - Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    print(f"   - Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"   - Epochs Trained: {len(history.history['accuracy'])}")
    print(f"   - Classes: {num_classes}")
    
    print(f"\nSaved Files:")
    print(f"   - Model: {MODEL_PATH}")
    print(f"   - Class indices: {CLASS_INDICES_PATH}")
    print(f"   - History: {HISTORY_PATH}")
    
    print("\n" + "="*80 + "\n")
    '''
    return model, history, test_accuracy


if __name__ == '__main__':
    print("\nDisease Detection CNN Trainer")
    print("="*80)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"{gpu}")
    else:
        print(f"No GPU found - training will use CPU (slower)")
    
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        model, history, accuracy = train_model()
        
        if model and accuracy >= 0.90:
            print(f"\nEXCELLENT! Achieved {accuracy*100:.2f}% accuracy ")
        elif model:
            print(f"\nGOOD! Achieved {accuracy*100:.2f}% accuracy")
        else:
            print(f"\nTraining failed!")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
