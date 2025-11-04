#!/usr/bin/env python3
"""
Quick Start Script for Improved Watermarking Model
===================================================

This script provides a simple interface to train the improved model.
Expected accuracy: 85-90% (vs original 50%)

Usage:
    python quick_start.py --data /path/to/images --epochs 20
"""

import argparse
from model_improved import train_model

def main():
    parser = argparse.ArgumentParser(description='Train improved watermarking model')
    
    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                      help='Path to image directory (jpg/png files)')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--payload-len', type=int, default=64,
                      help='Payload length in bits (default: 64)')
    parser.add_argument('--train-n', type=int, default=10000,
                      help='Number of training images (default: 10000)')
    parser.add_argument('--val-n', type=int, default=2000,
                      help='Number of validation images (default: 2000)')
    parser.add_argument('--test-n', type=int, default=2000,
                      help='Number of test images (default: 2000)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience (default: 5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("IMPROVED WATERMARKING MODEL TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Payload length: {args.payload_len} bits")
    print(f"  Dataset split: {args.train_n} train / {args.val_n} val / {args.test_n} test")
    print(f"  Early stopping patience: {args.patience}")
    print("\n" + "=" * 70)
    print("\nStarting training...\n")
    
    # Train the model
    encoder, decoder, history = train_model(
        root_images=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        payload_len=args.payload_len,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
        early_stop_patience=args.patience
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print("\nFiles saved:")
    print("  - best_model_checkpoint.pt  (model weights)")
    print("  - training_history.png      (training curves)")
    print("\nFinal validation accuracy from history:")
    if history['val_acc']:
        final_acc = history['val_acc'][-1] * 100
        best_acc = max(history['val_acc']) * 100
        print(f"  Final: {final_acc:.2f}%")
        print(f"  Best:  {best_acc:.2f}%")
        
        if best_acc >= 85:
            print(f"\n✅ SUCCESS! Achieved target accuracy (≥85%)")
        elif best_acc >= 75:
            print(f"\n⚠️  Close to target. Try training longer or adjusting hyperparameters.")
        else:
            print(f"\n❌ Below target. Check dataset quality and size.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
