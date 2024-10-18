import pandas as pd
import numpy as np
from typing import List, Dict, Any
import argparse
import os

class PredictionStringProcessor:
    def __init__(self, csv_path: str, save_path: str, conf_thr: float = 0.3):
        """
        csv_path: Path to CSV file containing model predictions
        save_path: Path to save processed output
        conf_thr: Confidence threshold for filtering predictions
        """
        self.csv_path = csv_path
        self.save_path = save_path
        self.conf_thr = conf_thr
    
    @staticmethod
    def parse_prediction_string(pred_string: str) -> List[Dict[str, Any]]:
        """
        Parse prediction string into list of detections
        Example input: "0 0.13577114 232.94168 667.66644 297.35226 745.04535"
        """
        if pd.isna(pred_string):
            return []
            
        values = pred_string.split()
        detections = []
        
        # Process 6 values at a time (category, confidence, x1, y1, x2, y2)
        for i in range(0, len(values), 6):
            if i + 6 <= len(values):
                detection = {
                    'category': int(values[i]),
                    'confidence': float(values[i + 1]),
                    'bbox': [
                        float(values[i + 2]),  # x1
                        float(values[i + 3]),  # y1
                        float(values[i + 4]),  # x2
                        float(values[i + 5])   # y2
                    ]
                }
                detections.append(detection)
        
        return detections

    def format_detection_to_string(self, detection: Dict[str, Any]) -> str:
        """Convert detection dictionary back to string format"""
        return (f"{detection['category']} "
                f"{detection['confidence']:.8f} "
                f"{detection['bbox'][0]:.5f} "
                f"{detection['bbox'][1]:.5f} "
                f"{detection['bbox'][2]:.5f} "
                f"{detection['bbox'][3]:.5f}")

    def process_single_row(self, pred_string: str) -> str:
        """Process a single prediction string and return filtered/formatted string"""
        # Parse the prediction string
        detections = self.parse_prediction_string(pred_string)
        
        # Filter detections based on confidence threshold
        filtered_detections = [det for det in detections if det['confidence'] >= self.conf_thr]
        
        # If no detections pass the threshold, return empty string
        if not filtered_detections:
            return ''
        
        # Convert filtered detections back to string format
        detection_strings = [self.format_detection_to_string(det) for det in filtered_detections]
        
        # Join all detections with space
        return ' '.join(detection_strings)

    def run(self):
        """Execute processing"""
        # Read CSV file
        df = pd.read_csv(self.csv_path)
        
        # Process each row
        df['PredictionString'] = df['PredictionString'].apply(self.process_single_row)
        
        # Save processed results
        df.to_csv(self.save_path, index=False)
        
        # Calculate statistics
        total_rows = len(df)
        rows_with_predictions = df['PredictionString'].str.len().gt(0).sum()
        
        return {
            'total_rows': total_rows,
            'rows_with_predictions': rows_with_predictions,
            'rows_filtered_out': total_rows - rows_with_predictions
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process PredictionString format data with confidence threshold filtering')
    
    parser.add_argument('--input_path', 
                       type=str, 
                       required=True,
                       help='Path to input CSV file containing PredictionString data')
    
    parser.add_argument('--output_path', 
                       type=str, 
                       help='Path to save processed CSV file (default: processed_{input_file})')
    
    parser.add_argument('--confidence_threshold', 
                       type=float, 
                       default=0.3,
                       help='Confidence threshold for filtering predictions (default: 0.3)')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # If output path is not specified, create default output path
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        input_filename = os.path.basename(args.input_path)
        args.output_path = os.path.join(input_dir, f"processed_{input_filename}")
    
    # Create processor instance
    processor = PredictionStringProcessor(
        csv_path=args.input_path,
        save_path=args.output_path,
        conf_thr=args.confidence_threshold
    )
    
    # Run processing
    try:
        print(f"Processing file: {args.input_path}")
        print(f"Confidence threshold: {args.confidence_threshold}")
        
        result = processor.run()
        
        print("\nProcessing completed successfully!")
        print(f"Total rows processed: {result['total_rows']}")
        print(f"Rows with predictions: {result['rows_with_predictions']}")
        print(f"Rows filtered out: {result['rows_filtered_out']}")
        print(f"\nResults saved to: {args.output_path}")
        
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
