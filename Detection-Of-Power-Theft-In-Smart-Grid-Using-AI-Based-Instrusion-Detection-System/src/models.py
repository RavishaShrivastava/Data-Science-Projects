"""
Deep Learning Models for Power Theft Detection
Includes CNN-LSTM, LSTM, and Neural Network architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib


class CNNLSTMModel:
    """
    CNN-LSTM Hybrid Model for Power Theft Detection
    Combines Convolutional layers for feature extraction with LSTM for temporal patterns
    """
    
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or {
            'cnn_filters': [64, 128],
            'lstm_units': [64, 32],
            'dropout': 0.3,
            'learning_rate': 0.001
        }
        self.model = None
        
    def build_model(self):
        """Build CNN-LSTM architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN layers for feature extraction
        x = inputs
        for filters in self.config['cnn_filters']:
            x = layers.Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(self.config['dropout'])(x)
        
        # LSTM layers for temporal patterns
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = (i < len(self.config['lstm_units']) - 1)
            x = layers.LSTM(units, return_sequences=return_sequences)(x)
            x = layers.Dropout(self.config['dropout'])(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class LSTMModel:
    """
    LSTM Model for Sequential Power Consumption Analysis
    """
    
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or {
            'units': [128, 64, 32],
            'dropout': 0.3,
            'learning_rate': 0.001
        }
        self.model = None
        
    def build_model(self):
        """Build LSTM architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        x = inputs
        for i, units in enumerate(self.config['units']):
            return_sequences = (i < len(self.config['units']) - 1)
            x = layers.LSTM(units, return_sequences=return_sequences)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config['dropout'])(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class NeuralNetworkModel:
    """
    Deep Neural Network for Power Theft Detection
    """
    
    def __init__(self, input_dim, config=None):
        self.input_dim = input_dim
        self.config = config or {
            'hidden_layers': [256, 128, 64, 32],
            'dropout': 0.3,
            'learning_rate': 0.001
        }
        self.model = None
        
    def build_model(self):
        """Build DNN architecture"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        x = inputs
        for units in self.config['hidden_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config['dropout'])(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class TraditionalMLModels:
    """
    Traditional Machine Learning Models for comparison
    Includes Random Forest, SVM, Decision Trees, Gradient Boosting
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print(f"\n{self.model_type.upper()} Performance:")
        print("=" * 50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    def save(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print(f"{'='*60}")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
    else:
        y_proba = model.predict(X_test).flatten()
    
    y_pred = (y_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Theft']))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    print("Power Theft Detection Models Module")
    print("Available models:")
    print("1. CNN-LSTM Model")
    print("2. LSTM Model")
    print("3. Neural Network Model")
    print("4. Traditional ML Models (Random Forest, SVM, etc.)")
