#!/usr/bin/env python3
"""
Visual verification - open and display a plot to check if OCEAN is included
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_plot_with_ocean():
    """Display a plot to visually verify OCEAN inclusion"""
    print("🖼️ Visual verification of OCEAN in plots...")
    
    # Look for a German Credit validity plot
    plot_path = 'img/VALIDITY/minor_deletion_German_Credit_validity_vs_values.png'
    
    if os.path.exists(plot_path):
        print(f"📊 Opening plot: {plot_path}")
        
        # Load and display the image
        img = mpimg.imread(plot_path)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('German Credit Minor Deletion Validity Plot - Check for OCEAN line', fontsize=14)
        plt.tight_layout()
        
        # Save a copy for inspection
        output_path = 'plot_verification.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved as {output_path}")
        print("🔍 Look for OCEAN line in the plot legend and graph!")
        
        # Check file size as indicator
        file_size = os.path.getsize(plot_path)
        print(f"📏 Plot file size: {file_size:,} bytes")
        
        if file_size > 200000:  # Larger files typically indicate more content
            print("✅ File size suggests comprehensive plot with multiple algorithms")
        else:
            print("⚠️ File size might indicate missing algorithms")
            
    else:
        print(f"❌ Plot not found at {plot_path}")
        print("🔍 Available plots:")
        
        validity_dir = 'img/VALIDITY'
        if os.path.exists(validity_dir):
            files = [f for f in os.listdir(validity_dir) if f.endswith('.png')]
            for f in files[:5]:
                print(f"   {f}")

if __name__ == "__main__":
    show_plot_with_ocean()
