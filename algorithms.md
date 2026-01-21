## Algorithms Used
### 1. Density Estimation
- **Method**: Grid-based pixel intensity analysis
- **Complexity**: O(n*m) where n,m = grid dimensions
- **Accuracy**: ~85% on test videos

### 2. Motion Analysis  
- **Method**: Farneback Optical Flow
- **Complexity**: O(width*height)
- **Output**: Motion vectors per frame

### 3. Anomaly Detection
- **Method**: Statistical Z-score analysis
- **Window**: 10-frame moving average
- **Threshold**: 3 standard deviations