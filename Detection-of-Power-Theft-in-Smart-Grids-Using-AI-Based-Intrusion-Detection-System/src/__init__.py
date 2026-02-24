"""
Power Theft Detection System
AI-Based Intrusion Detection for Smart Grids
"""

__version__ = '1.0.0'
__author__ = 'Ravisha'

def __getattr__(name):
    if name in {'DataPreprocessor', 'generate_sample_data'}:
        from .data_preprocessing import DataPreprocessor, generate_sample_data
        return {'DataPreprocessor': DataPreprocessor, 'generate_sample_data': generate_sample_data}[name]
    if name in {'CNNLSTMModel', 'LSTMModel', 'NeuralNetworkModel', 'TraditionalMLModels'}:
        from .models import CNNLSTMModel, LSTMModel, NeuralNetworkModel, TraditionalMLModels
        return {
            'CNNLSTMModel': CNNLSTMModel,
            'LSTMModel': LSTMModel,
            'NeuralNetworkModel': NeuralNetworkModel,
            'TraditionalMLModels': TraditionalMLModels,
        }[name]
    if name in {'IntrusionDetectionSystem', 'AnomalyDetector', 'RealTimeMonitor'}:
        from .intrusion_detection import IntrusionDetectionSystem, AnomalyDetector, RealTimeMonitor
        return {
            'IntrusionDetectionSystem': IntrusionDetectionSystem,
            'AnomalyDetector': AnomalyDetector,
            'RealTimeMonitor': RealTimeMonitor,
        }[name]
    if name == 'Visualizer':
        from .visualization import Visualizer
        return Visualizer
    if name == 'compute_risk_score':
        from .risk_scoring import compute_risk_score
        return compute_risk_score
    raise AttributeError(name)

__all__ = [
    'DataPreprocessor',
    "generate_sample_data",
    "compute_risk_score",
    "NeuralNetworkModel",
    "LSTMModel",
    "CNNLSTMModel",
    "TraditionalMLModels",
    'IntrusionDetectionSystem',
    'AnomalyDetector',
    'RealTimeMonitor',
    'Visualizer'
]
