"""
Intrusion Detection System for Power Theft
Real-time monitoring and alert generation
"""

import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import json


class IntrusionDetectionSystem:
    """
    AI-based Intrusion Detection System for Smart Grid Power Theft
    """
    
    def __init__(self, model, threshold=0.5, alert_config=None):
        self.model = model
        self.threshold = threshold
        self.alert_config = alert_config or {
            'high_risk': 0.8,
            'medium_risk': 0.5,
            'low_risk': 0.3
        }
        self.alert_history = deque(maxlen=1000)
        self.detection_log = []
        
    def detect_theft(self, consumption_data):
        """
        Detect potential power theft from consumption data
        
        Args:
            consumption_data: Preprocessed consumption features
            
        Returns:
            dict: Detection results with probability and risk level
        """
        # Get prediction probability
        risk_score = self.model.predict(consumption_data)
        
        if len(risk_score.shape) > 1:
            risk_score = risk_score.flatten()
        
        # Determine risk level
        risk_level = self._classify_risk(risk_score[0])
        
        # Create detection result
        result = {
            'timestamp': datetime.now().isoformat(),
            'risk_score': float(risk_score[0]),
            'risk_level': risk_level,
            'is_theft': risk_score[0] > self.threshold,
            'threshold': self.threshold
        }
        
        # Log detection
        self.detection_log.append(result)
        
        # Generate alert if needed
        if result['is_theft']:
            self._generate_alert(result)
        
        return result
    
    def batch_detect(self, consumption_batch):
        """
        Detect theft in batch of consumption data
        
        Args:
            consumption_batch: Array of preprocessed consumption features
            
        Returns:
            list: Detection results for each sample
        """
        risk_scores = self.model.predict(consumption_batch).flatten()
        
        results = []
        for i, score in enumerate(risk_scores):
            result = {
                'sample_id': i,
                'timestamp': datetime.now().isoformat(),
                'risk_score': float(score),
                'risk_level': self._classify_risk(score),
                'is_theft': score > self.threshold,
                'threshold': self.threshold
            }
            results.append(result)
            
            if result['is_theft']:
                self._generate_alert(result)
        
        return results
    
    def _classify_risk(self, probability):
        """Classify risk level based on theft probability"""
        if probability >= self.alert_config['high_risk']:
            return 'HIGH'
        elif probability >= self.alert_config['medium_risk']:
            return 'MEDIUM'
        elif probability >= self.alert_config['low_risk']:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _generate_alert(self, detection_result):
        """
        Generate alert for detected theft
        """
        alert = {
            'alert_id': len(self.alert_history) + 1,
            'timestamp': detection_result['timestamp'],
            'risk_level': detection_result['risk_level'],
            'risk_score': detection_result['risk_score'],
            'message': self._create_alert_message(detection_result),
            'status': 'ACTIVE',
            'acknowledged': False
        }
        
        self.alert_history.append(alert)
        
        # Print alert (in production, this would send to monitoring system)
        print(f"\n{'='*60}")
        print(f"⚠️  ALERT #{alert['alert_id']} - {alert['risk_level']} RISK")
        print(f"{'='*60}")
        print(f"Time: {alert['timestamp']}")
        print(f"Score: {alert['risk_score']:.2%}")
        print(f"Message: {alert['message']}")
        print(f"{'='*60}\n")
        
        return alert
    
    def _create_alert_message(self, detection_result):
        """Create human-readable alert message"""
        prob = detection_result['risk_score']
        risk = detection_result['risk_level']
        
        if risk == 'HIGH':
            return f"CRITICAL: High probability ({prob:.2%}) of power theft detected. Immediate investigation required."
        elif risk == 'MEDIUM':
            return f"WARNING: Moderate probability ({prob:.2%}) of power theft detected. Investigation recommended."
        else:
            return f"NOTICE: Low probability ({prob:.2%}) of abnormal consumption detected. Monitor closely."
    
    def get_active_alerts(self):
        """Get all active (unacknowledged) alerts"""
        return [alert for alert in self.alert_history if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['status'] = 'ACKNOWLEDGED'
                return True
        return False
    
    def get_detection_statistics(self):
        """Get statistics about detections"""
        if not self.detection_log:
            return {
                'total_detections': 0,
                'theft_detected': 0,
                'theft_rate': 0,
                'avg_probability': 0
            }
        
        total = len(self.detection_log)
        theft_count = sum(1 for d in self.detection_log if d['is_theft'])
        avg_prob = np.mean([d['risk_score'] for d in self.detection_log])
        
        return {
            'total_detections': total,
            'theft_detected': theft_count,
            'theft_rate': theft_count / total if total > 0 else 0,
            'avg_risk_score': float(avg_prob),
            'risk_distribution': self._get_risk_distribution()
        }
    
    def _get_risk_distribution(self):
        """Get distribution of risk levels"""
        risk_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NORMAL': 0}
        for detection in self.detection_log:
            risk_counts[detection['risk_level']] += 1
        return risk_counts
    
    def export_alerts(self, filepath):
        """Export alerts to JSON file"""
        alerts_data = {
            'export_time': datetime.now().isoformat(),
            'total_alerts': len(self.alert_history),
            'alerts': list(self.alert_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        print(f"Alerts exported to {filepath}")
    
    def export_detection_log(self, filepath):
        """Export detection log to CSV"""
        df = pd.DataFrame(self.detection_log)
        df.to_csv(filepath, index=False)
        print(f"Detection log exported to {filepath}")


class AnomalyDetector:
    """
    Statistical anomaly detection for consumption patterns
    """
    
    def __init__(self, window_size=168, threshold=3):
        """
        Args:
            window_size: Number of hours for rolling window (default: 1 week)
            threshold: Z-score threshold for anomaly detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_stats = {}
        
    def fit(self, consumption_data, user_id=None):
        """
        Fit baseline statistics for normal consumption
        
        Args:
            consumption_data: Historical consumption values
            user_id: Optional user identifier
        """
        key = user_id if user_id else 'global'
        
        self.baseline_stats[key] = {
            'mean': np.mean(consumption_data),
            'std': np.std(consumption_data),
            'median': np.median(consumption_data),
            'q25': np.percentile(consumption_data, 25),
            'q75': np.percentile(consumption_data, 75)
        }
        
        return self
    
    def detect_anomaly(self, consumption_value, user_id=None):
        """
        Detect if consumption value is anomalous
        
        Args:
            consumption_value: Current consumption value
            user_id: Optional user identifier
            
        Returns:
            dict: Anomaly detection result
        """
        key = user_id if user_id and user_id in self.baseline_stats else 'global'
        
        if key not in self.baseline_stats:
            return {
                'is_anomaly': False,
                'z_score': 0,
                'message': 'No baseline statistics available'
            }
        
        stats = self.baseline_stats[key]
        
        # Calculate z-score
        z_score = (consumption_value - stats['mean']) / (stats['std'] + 1e-6)
        
        # Check if anomalous
        is_anomaly = abs(z_score) > self.threshold
        
        # Determine anomaly type
        if is_anomaly:
            if z_score > 0:
                anomaly_type = 'HIGH_CONSUMPTION'
                message = f"Consumption {z_score:.2f} std deviations above normal"
            else:
                anomaly_type = 'LOW_CONSUMPTION'
                message = f"Consumption {abs(z_score):.2f} std deviations below normal (potential theft)"
        else:
            anomaly_type = 'NORMAL'
            message = 'Consumption within normal range'
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'z_score': float(z_score),
            'consumption_value': float(consumption_value),
            'baseline_mean': float(stats['mean']),
            'baseline_std': float(stats['std']),
            'message': message
        }
    
    def detect_sudden_drop(self, consumption_series, drop_threshold=0.3):
        """
        Detect sudden drops in consumption (indicator of meter tampering)
        
        Args:
            consumption_series: Time series of consumption values
            drop_threshold: Percentage drop to consider suspicious
            
        Returns:
            list: Detected drop events
        """
        drops = []
        
        for i in range(1, len(consumption_series)):
            prev_value = consumption_series[i-1]
            curr_value = consumption_series[i]
            
            if prev_value > 0:
                pct_change = (curr_value - prev_value) / prev_value
                
                if pct_change < -drop_threshold:
                    drops.append({
                        'index': i,
                        'previous_value': float(prev_value),
                        'current_value': float(curr_value),
                        'percent_drop': float(pct_change * 100),
                        'severity': 'HIGH' if pct_change < -0.5 else 'MEDIUM'
                    })
        
        return drops


class RealTimeMonitor:
    """
    Real-time monitoring system for power consumption
    """
    
    def __init__(self, ids_system, anomaly_detector=None):
        self.ids = ids_system
        self.anomaly_detector = anomaly_detector
        self.monitoring_active = False
        self.monitored_users = {}
        
    def start_monitoring(self, user_id=None):
        """Start monitoring for specific user or all users"""
        self.monitoring_active = True
        print(f"Monitoring started for {'user ' + str(user_id) if user_id else 'all users'}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        print("Monitoring stopped")
    
    def process_consumption_reading(self, user_id, consumption_value, features):
        """
        Process a single consumption reading
        
        Args:
            user_id: User identifier
            consumption_value: Raw consumption value
            features: Preprocessed feature vector
            
        Returns:
            dict: Processing result with theft detection and anomaly detection
        """
        if not self.monitoring_active:
            return {'error': 'Monitoring not active'}
        
        result = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'consumption_value': consumption_value
        }
        
        # Run intrusion detection
        theft_detection = self.ids.detect_theft(features)
        result['theft_detection'] = theft_detection
        
        # Run anomaly detection if available
        if self.anomaly_detector:
            anomaly_detection = self.anomaly_detector.detect_anomaly(consumption_value, user_id)
            result['anomaly_detection'] = anomaly_detection
        
        # Update user monitoring stats
        if user_id not in self.monitored_users:
            self.monitored_users[user_id] = {
                'readings_count': 0,
                'theft_detections': 0,
                'anomalies': 0
            }
        
        self.monitored_users[user_id]['readings_count'] += 1
        if theft_detection['is_theft']:
            self.monitored_users[user_id]['theft_detections'] += 1
        
        return result
    
    def get_user_statistics(self, user_id):
        """Get monitoring statistics for a specific user"""
        if user_id in self.monitored_users:
            stats = self.monitored_users[user_id]
            stats['theft_rate'] = stats['theft_detections'] / stats['readings_count'] if stats['readings_count'] > 0 else 0
            return stats
        return None
    
    def get_high_risk_users(self, threshold=0.3):
        """Get list of users with high theft detection rate"""
        high_risk = []
        
        for user_id, stats in self.monitored_users.items():
            if stats['readings_count'] > 0:
                theft_rate = stats['theft_detections'] / stats['readings_count']
                if theft_rate >= threshold:
                    high_risk.append({
                        'user_id': user_id,
                        'theft_rate': theft_rate,
                        'total_readings': stats['readings_count'],
                        'theft_detections': stats['theft_detections']
                    })
        
        # Sort by theft rate
        high_risk.sort(key=lambda x: x['theft_rate'], reverse=True)
        return high_risk


if __name__ == "__main__":
    print("Intrusion Detection System Module")
    print("Features:")
    print("- Real-time theft detection")
    print("- Anomaly detection")
    print("- Alert generation and management")
    print("- User monitoring and statistics")
