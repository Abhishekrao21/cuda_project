#!/usr/bin/env python3
"""
CUDA Histogram Plotting and Analysis Script
Visualizes and analyzes histogram results from the CUDA histogram computation.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

class HistogramAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.histograms = []
        self.summary_data = None

    def load_histograms(self):
        """Load histogram CSV files"""
        histogram_files = glob.glob(str(self.output_dir / "histogram_*.csv"))

        if not histogram_files:
            print(f"No histogram files found in {self.output_dir}")
            return False

        self.histograms = []
        for file_path in sorted(histogram_files):
            try:
                df = pd.read_csv(file_path)
                image_name = Path(file_path).stem
                df['image_name'] = image_name
                self.histograms.append(df)
                print(f"Loaded histogram: {Path(file_path).name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Total histograms loaded: {len(self.histograms)}")
        return len(self.histograms) > 0

    def load_summary(self):
        """Load summary statistics"""
        summary_file = self.output_dir / "histogram_summary.csv"
        if summary_file.exists():
            try:
                self.summary_data = pd.read_csv(summary_file)
                print(f"Loaded summary data for {len(self.summary_data)} images")
                return True
            except Exception as e:
                print(f"Error loading summary: {e}")
                return False
        else:
            print("Summary file not found")
            return False

    def plot_individual_histograms(self, max_plots=6):
        """Plot individual histograms for first few images"""
        if not self.histograms:
            print("No histograms to plot")
            return

        num_plots = min(len(self.histograms), max_plots)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(num_plots):
            df = self.histograms[i]
            axes[i].bar(df['Intensity'], df['Count'], width=1.0, alpha=0.7)
            axes[i].set_title(f'Histogram {i+1}\n{df["image_name"].iloc[0]}')
            axes[i].set_xlabel('Pixel Intensity')
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlim([0, 255])
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: individual_histograms.png")

    def plot_aggregate_histogram(self):
        """Plot aggregate histogram of all images"""
        if not self.histograms:
            print("No histograms to aggregate")
            return

        # Aggregate all histograms
        aggregate_counts = np.zeros(256)
        for df in self.histograms:
            aggregate_counts += df['Count'].values

        plt.figure(figsize=(12, 6))
        plt.bar(range(256), aggregate_counts, width=1.0, alpha=0.7, color='skyblue')
        plt.title(f'Aggregate Histogram (All {len(self.histograms)} Images)')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Total Frequency')
        plt.xlim([0, 255])
        plt.grid(True, alpha=0.3)

        # Add statistics text
        total_pixels = np.sum(aggregate_counts)
        mean_intensity = np.sum(range(256) * aggregate_counts) / total_pixels
        plt.text(0.7, 0.95, f'Total Pixels: {int(total_pixels):,}\nMean Intensity: {mean_intensity:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'aggregate_histogram.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: aggregate_histogram.png")

    def plot_summary_statistics(self):
        """Plot summary statistics"""
        if self.summary_data is None:
            print("No summary data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Mean intensity distribution
        axes[0, 0].hist(self.summary_data['Mean'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Distribution of Mean Intensities')
        axes[0, 0].set_xlabel('Mean Intensity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Standard deviation distribution
        axes[0, 1].hist(self.summary_data['Std'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Distribution of Standard Deviations')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Mean vs Std scatter plot
        axes[1, 0].scatter(self.summary_data['Mean'], self.summary_data['Std'], 
                          alpha=0.6, color='red')
        axes[1, 0].set_title('Mean vs Standard Deviation')
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].grid(True, alpha=0.3)

        # Intensity range (max - min)
        intensity_range = self.summary_data['Max'] - self.summary_data['Min']
        axes[1, 1].hist(intensity_range, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_title('Distribution of Intensity Ranges')
        axes[1, 1].set_xlabel('Intensity Range (Max - Min)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: summary_statistics.png")

    def detect_anomalies(self, threshold=2.0):
        """Detect potential anomalies based on statistical outliers"""
        if self.summary_data is None:
            print("No summary data available for anomaly detection")
            return

        print("\n=== Anomaly Detection Results ===")

        # Z-score based anomaly detection
        for column in ['Mean', 'Std']:
            mean_val = self.summary_data[column].mean()
            std_val = self.summary_data[column].std()
            z_scores = np.abs((self.summary_data[column] - mean_val) / std_val)

            anomalies = self.summary_data[z_scores > threshold]
            if len(anomalies) > 0:
                print(f"\nAnomalies in {column} (threshold={threshold}):")
                for idx, row in anomalies.iterrows():
                    z_score = z_scores.iloc[idx]
                    print(f"  {row['Image']}: {column}={row[column]:.2f} (z-score={z_score:.2f})")
            else:
                print(f"No anomalies detected in {column}")

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        if self.summary_data is None:
            print("Cannot generate report without summary data")
            return

        report_file = self.output_dir / "analysis_report.txt"

        with open(report_file, 'w') as f:
            f.write("CUDA Histogram Analysis Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Dataset Overview:\n")
            f.write(f"  Total Images Processed: {len(self.summary_data)}\n")
            f.write(f"  Total Pixels: {self.summary_data['Total_Pixels'].sum():,}\n\n")

            f.write("Statistical Summary:\n")
            f.write(f"  Mean Intensity - Avg: {self.summary_data['Mean'].mean():.2f}, "
                   f"Range: [{self.summary_data['Mean'].min():.2f}, {self.summary_data['Mean'].max():.2f}]\n")
            f.write(f"  Std Deviation - Avg: {self.summary_data['Std'].mean():.2f}, "
                   f"Range: [{self.summary_data['Std'].min():.2f}, {self.summary_data['Std'].max():.2f}]\n")

            # Find extremes
            brightest_img = self.summary_data.loc[self.summary_data['Mean'].idxmax()]
            darkest_img = self.summary_data.loc[self.summary_data['Mean'].idxmin()]
            highest_contrast = self.summary_data.loc[self.summary_data['Std'].idxmax()]
            lowest_contrast = self.summary_data.loc[self.summary_data['Std'].idxmin()]

            f.write(f"\nImage Characteristics:\n")
            f.write(f"  Brightest Image: {brightest_img['Image']} (mean={brightest_img['Mean']:.2f})\n")
            f.write(f"  Darkest Image: {darkest_img['Image']} (mean={darkest_img['Mean']:.2f})\n")
            f.write(f"  Highest Contrast: {highest_contrast['Image']} (std={highest_contrast['Std']:.2f})\n")
            f.write(f"  Lowest Contrast: {lowest_contrast['Image']} (std={lowest_contrast['Std']:.2f})\n")

        print(f"Generated analysis report: {report_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_histograms.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        sys.exit(1)

    print("CUDA Histogram Analysis and Plotting")
    print("=" * 40)

    analyzer = HistogramAnalyzer(output_dir)

    # Load data
    if not analyzer.load_histograms():
        print("Failed to load histogram data")
        sys.exit(1)

    analyzer.load_summary()

    # Generate plots
    print("\nGenerating plots...")
    analyzer.plot_individual_histograms()
    analyzer.plot_aggregate_histogram()

    if analyzer.summary_data is not None:
        analyzer.plot_summary_statistics()
        analyzer.detect_anomalies()
        analyzer.generate_report()

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
