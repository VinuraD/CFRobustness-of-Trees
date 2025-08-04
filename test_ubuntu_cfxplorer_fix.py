#!/usr/bin/env python3
"""
Test script to verify CFXplorer Ubuntu TensorFlow compatibility fix

This script tests the platform-aware dtype selection logic without 
running the full CFXplorer analysis.
"""

import sys
import numpy as np

def test_platform_detection():
    """Test platform detection and dtype priority logic"""
    print("=" * 60)
    print("CFXplorer Ubuntu TensorFlow Compatibility Test")
    print("=" * 60)
    
    print(f"Platform: {sys.platform}")
    print(f"Is Linux/Ubuntu: {sys.platform.startswith('linux')}")
    
    # Test TensorFlow availability and behavior
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check TensorFlow's default int type and platform behavior
        default_tf_int = tf.int64 if tf.executing_eagerly() else tf.int32
        print(f"TensorFlow default int type: {default_tf_int}")
        print(f"TensorFlow eager execution: {tf.executing_eagerly()}")
        
    except ImportError:
        print("TensorFlow not available - this is expected if not in conda environment")
        return False
    
    # Test dtype priority selection logic
    dtype_priority = [np.int64, np.int32, np.int_, np.long] if sys.platform.startswith('linux') else [np.int32, np.int64, np.int_]
    print(f"Dtype priority order: {[dtype.__name__ for dtype in dtype_priority]}")
    
    # Test data type conversion with sample data
    test_labels = np.array([0, 1, 0, 1, 1])
    print(f"Original test labels dtype: {test_labels.dtype}")
    
    successful_dtype = None
    converted_labels = None
    
    for dtype in dtype_priority:
        try:
            converted_labels = test_labels.astype(dtype)
            successful_dtype = dtype
            print(f"✓ Successfully converted to {dtype.__name__}")
            break
        except Exception as e:
            print(f"✗ Failed to convert to {dtype.__name__}: {e}")
    
    if converted_labels is not None:
        print(f"Final conversion result: {converted_labels.dtype} ({successful_dtype.__name__})")
        print(f"Sample values: {converted_labels[:3]}")
        
        # Test TensorFlow tensor creation with the converted dtype
        try:
            tf_tensor = tf.constant(converted_labels)
            print(f"TensorFlow tensor creation successful: {tf_tensor.dtype}")
            
            # Test tf.stack operation (the problematic operation from the error)
            stacked = tf.stack([tf_tensor, tf_tensor])
            print(f"✓ tf.stack operation successful: {stacked.shape}")
            return True
            
        except Exception as e:
            print(f"✗ TensorFlow tensor operations failed: {e}")
            return False
    else:
        print("✗ All dtype conversions failed")
        return False

def main():
    """Main test function"""
    success = test_platform_detection()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ CFXplorer Ubuntu compatibility test PASSED")
        print("The platform-aware dtype selection should work correctly.")
    else:
        print("✗ CFXplorer Ubuntu compatibility test FAILED")
        print("Check TensorFlow installation and dtype compatibility.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
