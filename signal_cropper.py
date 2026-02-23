"""
Signal Cropper with FFT Analysis and SNR Calculation
A GUI application for cropping signals and analyzing them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from scipy.signal import butter, sosfiltfilt
from scipy.stats import entropy
import os


class SignalCropperApp:
    def __init__(self, root, signal=None, sample_rate=60):
        self.root = root
        self.root.title("Signal Cropper with FFT Analysis")
        self.root.geometry("1400x1000")
        
        # Signal data
        if signal is None:
            # Generate a demo signal if none provided
            self.sample_rate = sample_rate
            self.signal = self.generate_demo_signal()
        else:
            self.signal = signal
            self.sample_rate = sample_rate
        
        self.time = np.arange(len(self.signal)) / self.sample_rate
        
        # Crop storage
        self.crops = []  # List of tuples: (start_idx, end_idx, name)
        self.crop_patches = []  # Visual patches on the plot
        self.crop_counter = 1
        
        # Current selection
        self.current_selection = None
        
        self.setup_ui()
        self.plot_signal()
        
    def generate_demo_signal(self):
        """Generate a demo signal with multiple frequency components and noise."""
        duration = 10  # seconds
        n_samples = duration * self.sample_rate
        t = np.linspace(0, duration, n_samples)  # 10 seconds for better low-freq resolution
        
        # Multiple frequency components including 1-4 Hz range
        signal = (
            1.5 * np.sin(2 * np.pi * 1.5 * t) +     # 1.5 Hz (in passband)
            1.0 * np.sin(2 * np.pi * 2.5 * t) +     # 2.5 Hz (in passband)
            0.8 * np.sin(2 * np.pi * 3.5 * t) +     # 3.5 Hz (in passband)
            0.5 * np.sin(2 * np.pi * 10 * t) +      # 10 Hz (will be filtered)
            0.3 * np.sin(2 * np.pi * 25 * t) +      # 25 Hz (will be filtered)
            0.2 * np.sin(2 * np.pi * 50 * t)        # 50 Hz (will be filtered)
        )
        
        # Add DC offset (will be removed by preprocessing)
        signal += 2.0
        
        # Add some noise
        noise = 0.3 * np.random.randn(len(t))
        signal += noise
        
        # Add a transient in the middle (sample_rate independent)
        trans_start = int(n_samples * 0.4)  # 40% into signal
        trans_end = int(n_samples * 0.5)    # 50% into signal
        trans_len = trans_end - trans_start
        if trans_len > 0:
            signal[trans_start:trans_end] += 1.5 * np.exp(-0.005 * np.arange(trans_len)) * np.sin(2 * np.pi * 2 * t[trans_start:trans_end])
        
        return signal
    
    def setup_ui(self):
        """Setup the main UI components."""
        # Main container with vertical PanedWindow for resizable top/bottom split
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Top frame for signal plot and controls
        self.top_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.top_frame, weight=1)
        
        # Left panel - Signal plot
        self.plot_frame = ttk.LabelFrame(self.top_frame, text="Original Signal (Click and drag to select crop region)", padding="5")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create figure for signal
        self.fig_signal, self.ax_signal = plt.subplots(figsize=(10, 3))
        self.fig_signal.tight_layout()
        
        self.canvas_signal = FigureCanvasTkAgg(self.fig_signal, master=self.plot_frame)
        self.canvas_signal.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar for signal plot
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas_signal, self.toolbar_frame)
        self.toolbar.update()
        
        # Right panel - Controls (scrollable)
        self.control_outer_frame = ttk.LabelFrame(self.top_frame, text="Controls", padding="5")
        self.control_outer_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Create canvas and scrollbar for scrolling
        self.control_canvas = tk.Canvas(self.control_outer_frame, width=200, highlightthickness=0)
        self.control_scrollbar = ttk.Scrollbar(self.control_outer_frame, orient=tk.VERTICAL, 
                                                command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create frame inside canvas for controls
        self.control_frame = ttk.Frame(self.control_canvas, padding="5")
        self.control_canvas_window = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor=tk.NW)
        
        # Configure scroll region when frame size changes
        def configure_scroll_region(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        self.control_frame.bind("<Configure>", configure_scroll_region)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.control_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Selection info
        ttk.Label(self.control_frame, text="Current Selection:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.selection_label = ttk.Label(self.control_frame, text="None", wraplength=180)
        self.selection_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Add crop button
        self.add_crop_btn = ttk.Button(self.control_frame, text="Add Crop", command=self.add_crop, state=tk.DISABLED)
        self.add_crop_btn.pack(fill=tk.X, pady=2)
        
        # Clear selection button
        self.clear_sel_btn = ttk.Button(self.control_frame, text="Clear Selection", command=self.clear_selection)
        self.clear_sel_btn.pack(fill=tk.X, pady=2)
        
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Crops list
        ttk.Label(self.control_frame, text="Saved Crops:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Listbox for crops
        self.crops_listbox = tk.Listbox(self.control_frame, height=8, width=25)
        self.crops_listbox.pack(fill=tk.X, pady=5)
        
        # Delete selected crop
        self.delete_crop_btn = ttk.Button(self.control_frame, text="Delete Selected Crop", command=self.delete_crop)
        self.delete_crop_btn.pack(fill=tk.X, pady=2)
        
        # Clear all crops
        self.clear_all_btn = ttk.Button(self.control_frame, text="Clear All Crops", command=self.clear_all_crops)
        self.clear_all_btn.pack(fill=tk.X, pady=2)
        
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Analyze button
        self.analyze_btn = ttk.Button(self.control_frame, text="Analyze Crops", command=self.analyze_crops, 
                                       style='Accent.TButton')
        self.analyze_btn.pack(fill=tk.X, pady=5)
        
        # Save results button
        self.save_btn = ttk.Button(self.control_frame, text="Save Results", command=self.save_results)
        self.save_btn.pack(fill=tk.X, pady=2)
        
        # Load signal button
        self.load_btn = ttk.Button(self.control_frame, text="Load Signal (CSV/NPY)", command=self.load_signal)
        self.load_btn.pack(fill=tk.X, pady=2)
        
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sample rate entry (fixed at 60 Hz)
        ttk.Label(self.control_frame, text="Sample Rate (Hz):").pack(anchor=tk.W)
        self.sample_rate_var = tk.StringVar(value="60")
        self.sample_rate_entry = ttk.Entry(self.control_frame, textvariable=self.sample_rate_var, width=15, state='readonly')
        self.sample_rate_entry.pack(anchor=tk.W, pady=(0, 5))

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Auto-Find Best OPA section
        ttk.Label(self.control_frame, text="Auto-Find Best OPA:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Window length entry
        ttk.Label(self.control_frame, text="Window Length (s):").pack(anchor=tk.W)
        self.window_length_var = tk.StringVar(value="5.0")
        self.window_length_entry = ttk.Entry(self.control_frame, textvariable=self.window_length_var, width=15)
        self.window_length_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Top K entry
        ttk.Label(self.control_frame, text="Top K Windows:").pack(anchor=tk.W)
        self.top_k_var = tk.StringVar(value="5")
        self.top_k_entry = ttk.Entry(self.control_frame, textvariable=self.top_k_var, width=15)
        self.top_k_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Stride entry (in frames)
        ttk.Label(self.control_frame, text="Stride (frames):").pack(anchor=tk.W)
        self.stride_var = tk.StringVar(value="1")
        self.stride_entry = ttk.Entry(self.control_frame, textvariable=self.stride_var, width=15)
        self.stride_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Bandpass filter checkbox (for pre-filtered signals like opa_lateral_percent)
        self.apply_bandpass_var = tk.BooleanVar(value=True)
        self.apply_bandpass_check = ttk.Checkbutton(
            self.control_frame, 
            text="Apply 1-4Hz Bandpass", 
            variable=self.apply_bandpass_var
        )
        self.apply_bandpass_check.pack(anchor=tk.W, pady=(0, 5))
        
        # Auto-find button
        self.auto_find_btn = ttk.Button(self.control_frame, text="Auto-Find Best OPA", 
                                         command=self.auto_find_best_opa)
        self.auto_find_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Batch Folder Analysis section
        ttk.Label(self.control_frame, text="Batch Folder Analysis:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Signal filename entry
        ttk.Label(self.control_frame, text="Signal file (no .npy):").pack(anchor=tk.W)
        self.batch_signal_var = tk.StringVar(value="opa_lateral_percent")
        self.batch_signal_entry = ttk.Entry(self.control_frame, textvariable=self.batch_signal_var, width=20)
        self.batch_signal_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Subfolder entry
        ttk.Label(self.control_frame, text="Subfolder:").pack(anchor=tk.W)
        self.batch_subfolder_var = tk.StringVar(value="left_eye")
        self.batch_subfolder_entry = ttk.Entry(self.control_frame, textvariable=self.batch_subfolder_var, width=15)
        self.batch_subfolder_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Batch Top K entry
        ttk.Label(self.control_frame, text="Top K per patient:").pack(anchor=tk.W)
        self.batch_top_k_var = tk.StringVar(value="5")
        self.batch_top_k_entry = ttk.Entry(self.control_frame, textvariable=self.batch_top_k_var, width=10)
        self.batch_top_k_entry.pack(anchor=tk.W, pady=(0, 5))
        
        # Analyze folder button
        self.analyze_folder_btn = ttk.Button(self.control_frame, text="Analyze Folder...", 
                                              command=self.analyze_folder)
        self.analyze_folder_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Narrow Band-Pass Filter section
        ttk.Label(self.control_frame, text="Narrow Band-Pass Filter:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)

        # Crop selector
        ttk.Label(self.control_frame, text="Select Crop:").pack(anchor=tk.W)
        self.filter_crop_var = tk.StringVar()
        self.filter_crop_combo = ttk.Combobox(self.control_frame, textvariable=self.filter_crop_var,
                                               state='readonly', width=22)
        self.filter_crop_combo.pack(fill=tk.X, pady=(0, 5))

        # f0 entry
        ttk.Label(self.control_frame, text="Center Freq f0 (Hz):").pack(anchor=tk.W)
        self.f0_var = tk.StringVar(value="2.0")
        self.f0_entry = ttk.Entry(self.control_frame, textvariable=self.f0_var, width=15)
        self.f0_entry.pack(anchor=tk.W, pady=(0, 5))

        # Bandwidth entry
        ttk.Label(self.control_frame, text="Bandwidth (Hz) [0.1-0.2]:").pack(anchor=tk.W)
        self.bw_var = tk.StringVar(value="0.15")
        self.bw_entry = ttk.Entry(self.control_frame, textvariable=self.bw_var, width=15)
        self.bw_entry.pack(anchor=tk.W, pady=(0, 5))

        # Apply filter button
        self.apply_filter_btn = ttk.Button(self.control_frame, text="Apply Narrow BPF",
                                            command=self.apply_narrow_bandpass, state=tk.DISABLED)
        self.apply_filter_btn.pack(fill=tk.X, pady=2)

        # Bottom frame for analysis results
        self.bottom_frame = ttk.LabelFrame(self.paned_window, text="Analysis Results", padding="5")
        self.paned_window.add(self.bottom_frame, weight=2)
        
        # Scrollable canvas for results
        self.results_canvas = tk.Canvas(self.bottom_frame)
        self.results_scrollbar_y = ttk.Scrollbar(self.bottom_frame, orient=tk.VERTICAL, command=self.results_canvas.yview)
        self.results_scrollbar_x = ttk.Scrollbar(self.bottom_frame, orient=tk.HORIZONTAL, command=self.results_canvas.xview)
        self.results_inner_frame = ttk.Frame(self.results_canvas)
        
        self.results_inner_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.results_inner_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar_y.set, xscrollcommand=self.results_scrollbar_x.set)
        
        self.results_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Placeholder label
        self.placeholder_label = ttk.Label(self.results_inner_frame, 
                                           text="Add crops and click 'Analyze Crops' to see results here",
                                           font=('Arial', 12))
        self.placeholder_label.pack(pady=50)
        
    def plot_signal(self):
        """Plot the original signal with span selector."""
        self.ax_signal.clear()
        self.ax_signal.plot(self.time, self.signal, 'b-', linewidth=0.8, label='Signal')
        self.ax_signal.set_xlabel('Time (s)')
        self.ax_signal.set_ylabel('Amplitude')
        self.ax_signal.set_title('Original Signal')
        self.ax_signal.grid(True, alpha=0.3)
        self.ax_signal.legend(loc='upper right')
        
        # Redraw crop patches
        self.redraw_crop_patches()
        
        # Setup span selector
        self.span_selector = SpanSelector(
            self.ax_signal,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='green'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        self.canvas_signal.draw()
        
    def on_select(self, xmin, xmax):
        """Handle span selection."""
        # Convert time to indices
        start_idx = max(0, int(xmin * self.sample_rate))
        end_idx = min(len(self.signal), int(xmax * self.sample_rate))
        
        if end_idx - start_idx < 10:
            self.selection_label.config(text="Selection too small")
            self.add_crop_btn.config(state=tk.DISABLED)
            return
        
        self.current_selection = (start_idx, end_idx)
        
        # Update label
        duration = (end_idx - start_idx) / self.sample_rate
        self.selection_label.config(
            text=f"Time: {xmin:.3f}s - {xmax:.3f}s\n"
                 f"Duration: {duration:.3f}s\n"
                 f"Samples: {end_idx - start_idx}"
        )
        self.add_crop_btn.config(state=tk.NORMAL)
        
    def add_crop(self):
        """Add current selection as a crop."""
        if self.current_selection is None:
            return
        
        start_idx, end_idx = self.current_selection
        crop_name = f"Crop {self.crop_counter}"
        self.crops.append((start_idx, end_idx, crop_name))
        self.crop_counter += 1
        
        # Update listbox
        start_time = start_idx / self.sample_rate
        end_time = end_idx / self.sample_rate
        self.crops_listbox.insert(tk.END, f"{crop_name}: {start_time:.3f}s - {end_time:.3f}s")
        
        # Draw patch on signal
        self.redraw_crop_patches()
        
        # Clear selection
        self.clear_selection()
        
        messagebox.showinfo("Crop Added", f"{crop_name} has been added!")
        
    def redraw_crop_patches(self):
        """Redraw all crop patches on the signal plot."""
        # Remove old patches safely (they may have already been removed if axes was cleared)
        for patch in self.crop_patches:
            try:
                patch.remove()
            except (NotImplementedError, ValueError):
                # Patch was already removed or can't be removed
                pass
        self.crop_patches.clear()
        
        # Draw new patches
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (start_idx, end_idx, name) in enumerate(self.crops):
            start_time = start_idx / self.sample_rate
            end_time = end_idx / self.sample_rate
            color = colors[i % len(colors)]
            
            patch = self.ax_signal.axvspan(start_time, end_time, alpha=0.2, color=color, label=name)
            self.crop_patches.append(patch)
            
            # Add text label
            mid_time = (start_time + end_time) / 2
            y_pos = self.ax_signal.get_ylim()[1] * 0.9
            text = self.ax_signal.text(mid_time, y_pos, name, ha='center', fontsize=8, 
                                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
            self.crop_patches.append(text)
        
        self.canvas_signal.draw()
        
    def clear_selection(self):
        """Clear current selection."""
        self.current_selection = None
        self.selection_label.config(text="None")
        self.add_crop_btn.config(state=tk.DISABLED)
        
    def delete_crop(self):
        """Delete selected crop from list."""
        selection = self.crops_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a crop to delete.")
            return
        
        idx = selection[0]
        self.crops.pop(idx)
        self.crops_listbox.delete(idx)
        self.redraw_crop_patches()
        
    def clear_all_crops(self):
        """Clear all crops."""
        if self.crops:
            if messagebox.askyesno("Confirm", "Are you sure you want to clear all crops?"):
                self.crops.clear()
                self.crops_listbox.delete(0, tk.END)
                self.redraw_crop_patches()
                
    def calculate_snr(self, signal):
        """Calculate Signal-to-Noise Ratio using different methods."""
        # Method 1: RMS-based SNR (assuming noise is the high-frequency component)
        # Apply a simple moving average to estimate the "signal"
        window_size = min(50, len(signal) // 10)
        if window_size < 3:
            window_size = 3
        
        # Estimate signal using moving average
        kernel = np.ones(window_size) / window_size
        signal_estimate = np.convolve(signal, kernel, mode='same')
        
        # Estimate noise as the difference
        noise_estimate = signal - signal_estimate
        
        # Calculate RMS
        signal_rms = np.sqrt(np.mean(signal_estimate ** 2))
        noise_rms = np.sqrt(np.mean(noise_estimate ** 2))
        
        if noise_rms == 0:
            snr_db = float('inf')
        else:
            snr_db = 20 * np.log10(signal_rms / noise_rms)
        
        # Method 2: Peak signal to noise floor ratio
        signal_power = np.mean(signal ** 2)
        peak_amplitude = np.max(np.abs(signal))
        
        return {
            'snr_db': snr_db,
            'signal_rms': signal_rms,
            'noise_rms': noise_rms,
            'signal_power': signal_power,
            'peak_amplitude': peak_amplitude,
            'dynamic_range_db': 20 * np.log10(peak_amplitude / noise_rms) if noise_rms > 0 else float('inf')
        }
    
    def compute_fft(self, signal):
        """Compute FFT of the signal."""
        n = len(signal)
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(n)
        windowed_signal = signal * window
        
        # Compute FFT
        fft_result = np.fft.fft(windowed_signal)
        fft_magnitude = np.abs(fft_result[:n // 2])
        fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)  # Add small value to avoid log(0)
        
        # Frequency axis
        frequencies = np.fft.fftfreq(n, 1 / self.sample_rate)[:n // 2]
        
        return frequencies, fft_magnitude, fft_magnitude_db
    
    def preprocess_signal(self, signal, lowcut=0.5, highcut=4.0):
        """
        Preprocess signal by:
        1. Removing DC component (subtract mean)
        2. Applying zero-phase bandpass filter (1-4 Hz by default)
        
        Parameters:
        -----------
        signal : array-like
            Input signal to preprocess
        lowcut : float
            Low cutoff frequency in Hz (default: 1 Hz)
        highcut : float
            High cutoff frequency in Hz (default: 4 Hz)
            
        Returns:
        --------
        preprocessed_signal : array
            Preprocessed signal
        """
        # Step 1: Remove DC component by subtracting mean
        signal_no_dc = signal - np.mean(signal)
        
        # Step 2: Design Butterworth bandpass filter
        nyquist = self.sample_rate / 2.0
        
        # Check if filter frequencies are valid
        if lowcut >= nyquist or highcut >= nyquist:
            print(f"Warning: Filter cutoff frequencies ({lowcut}-{highcut} Hz) exceed Nyquist frequency ({nyquist} Hz). Adjusting...")
            lowcut = min(lowcut, nyquist * 0.9)
            highcut = min(highcut, nyquist * 0.95)
        
        if lowcut >= highcut:
            print("Warning: Invalid filter range. Returning DC-removed signal only.")
            return signal_no_dc
        
        # Normalize frequencies
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design 4th order Butterworth bandpass filter
        order = 4
        sos = butter(order, [low, high], btype='band', output='sos')
        
        # Step 3: Apply zero-phase filtering using sosfiltfilt
        # sosfiltfilt applies the filter twice (forward and backward) for zero phase distortion
        try:
            # Ensure signal is long enough for filtering
            min_padlen = 3 * max(len(sos) * 2, 1)
            if len(signal_no_dc) <= min_padlen:
                print("Warning: Signal too short for filtering. Returning DC-removed signal only.")
                return signal_no_dc
            
            preprocessed_signal = sosfiltfilt(sos, signal_no_dc)
        except Exception as e:
            print(f"Warning: Filtering failed ({e}). Returning DC-removed signal only.")
            return signal_no_dc
        
        return preprocessed_signal
    
    def analyze_crops(self):
        """Analyze all crops and display results."""
        if not self.crops:
            messagebox.showwarning("Warning", "No crops to analyze. Please add some crops first.")
            return
        
        # Clear previous results
        for widget in self.results_inner_frame.winfo_children():
            widget.destroy()
        
        # Create figure for all crops
        num_crops = len(self.crops)
        
        # Create a figure with subplots for each crop (4 columns: raw, preprocessed, FFT linear, FFT dB)
        fig, axes = plt.subplots(num_crops, 4, figsize=(18, 4 * num_crops))
        if num_crops == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Crop Analysis Results (Preprocessed: DC removed + 1-4 Hz Bandpass Filter)', fontsize=14, fontweight='bold')
        
        results_data = []
        
        for i, (start_idx, end_idx, name) in enumerate(self.crops):
            # Extract cropped signal
            cropped_signal_raw = self.signal[start_idx:end_idx]
            cropped_time = self.time[start_idx:end_idx] - self.time[start_idx]
            
            # Apply preprocessing: Remove DC + 1-4 Hz bandpass filter
            cropped_signal = self.preprocess_signal(cropped_signal_raw, lowcut=0.5, highcut=4.0)
            
            # Calculate SNR on preprocessed signal
            snr_data = self.calculate_snr(cropped_signal)
            
            # Compute FFT on preprocessed signal
            frequencies, fft_magnitude, fft_magnitude_db = self.compute_fft(cropped_signal)
            
            # Store results
            results_data.append({
                'name': name,
                'start_time': start_idx / self.sample_rate,
                'end_time': end_idx / self.sample_rate,
                'duration': len(cropped_signal) / self.sample_rate,
                'samples': len(cropped_signal),
                'snr': snr_data
            })
            
            # Plot 1: Raw cropped signal
            ax_raw = axes[i, 0]
            ax_raw.plot(cropped_time, cropped_signal_raw, 'b-', linewidth=0.8)
            ax_raw.set_xlabel('Time (s)')
            ax_raw.set_ylabel('Amplitude')
            ax_raw.set_title(f'{name} - Raw Signal')
            ax_raw.grid(True, alpha=0.3)
            
            # Plot 2: Preprocessed cropped signal
            ax_signal = axes[i, 1]
            ax_signal.plot(cropped_time, cropped_signal, 'b-', linewidth=0.8)
            ax_signal.set_xlabel('Time (s)')
            ax_signal.set_ylabel('Amplitude')
            ax_signal.set_title(f'{name} - Preprocessed (DC removed + 1-4Hz BP)')
            ax_signal.grid(True, alpha=0.3)
            
            # Add SNR info as text box
            textstr = f'SNR: {snr_data["snr_db"]:.2f} dB\nRMS: {snr_data["signal_rms"]:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_signal.text(0.02, 0.98, textstr, transform=ax_signal.transAxes, fontsize=8,
                          verticalalignment='top', bbox=props)
            
            # Plot 3: FFT (linear scale)
            ax_fft_linear = axes[i, 2]
            ax_fft_linear.plot(frequencies, fft_magnitude, 'r-', linewidth=0.8)
            ax_fft_linear.set_xlabel('Frequency (Hz)')
            ax_fft_linear.set_ylabel('Magnitude')
            ax_fft_linear.set_title(f'{name} - FFT (Linear)')
            ax_fft_linear.grid(True, alpha=0.3)
            ax_fft_linear.set_xlim(0, min(10, self.sample_rate / 2))  # Focus on 0-10 Hz for 1-4 Hz filter
            
            # Plot 4: FFT (dB scale)
            ax_fft_db = axes[i, 3]
            ax_fft_db.plot(frequencies, fft_magnitude_db, 'g-', linewidth=0.8)
            ax_fft_db.set_xlabel('Frequency (Hz)')
            ax_fft_db.set_ylabel('Magnitude (dB)')
            ax_fft_db.set_title(f'{name} - FFT (dB)')
            ax_fft_db.grid(True, alpha=0.3)
            ax_fft_db.set_xlim(0, min(10, self.sample_rate / 2))  # Focus on 0-10 Hz for 1-4 Hz filter
            
            # Find and annotate dominant frequencies
            peak_indices = self.find_peaks(fft_magnitude)
            for peak_idx in peak_indices[:5]:  # Show top 5 peaks
                if fft_magnitude[peak_idx] > np.max(fft_magnitude) * 0.1:  # Only show significant peaks
                    ax_fft_linear.annotate(f'{frequencies[peak_idx]:.1f} Hz',
                                          xy=(frequencies[peak_idx], fft_magnitude[peak_idx]),
                                          fontsize=7, ha='center')
        
        fig.tight_layout()
        
        # Embed figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.results_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add summary table
        self.create_summary_table(results_data)
        
        # Store results for saving
        self.analysis_results = results_data
        self.results_figure = fig

        # Enable narrow band-pass filter controls
        crop_names = [name for _, _, name in self.crops]
        self.filter_crop_combo['values'] = crop_names
        if crop_names:
            self.filter_crop_combo.current(0)
        self.apply_filter_btn.config(state=tk.NORMAL)

    def apply_narrow_bandpass(self):
        """Apply a narrow band-pass filter to the selected crop and display the result."""
        # Get selected crop
        crop_name = self.filter_crop_var.get()
        if not crop_name:
            messagebox.showwarning("Warning", "Please select a crop to filter.")
            return

        # Find the crop
        crop_idx = None
        for i, (start_idx, end_idx, name) in enumerate(self.crops):
            if name == crop_name:
                crop_idx = i
                break

        if crop_idx is None:
            messagebox.showwarning("Warning", "Selected crop not found.")
            return

        # Get f0 and bandwidth
        try:
            f0 = float(self.f0_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid center frequency (f0). Please enter a number.")
            return

        try:
            bw = float(self.bw_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid bandwidth. Please enter a number.")
            return

        if f0 <= 0:
            messagebox.showerror("Error", "Center frequency f0 must be positive.")
            return

        if bw < 0.01 or bw > 10:
            messagebox.showerror("Error", "Bandwidth should be between 0.01 and 10 Hz.")
            return

        lowcut = f0 - bw / 2
        highcut = f0 + bw / 2

        if lowcut <= 0:
            messagebox.showerror("Error",
                                 f"f0 - bandwidth/2 = {lowcut:.3f} Hz is not positive.\n"
                                 f"Increase f0 or decrease bandwidth.")
            return

        nyquist = self.sample_rate / 2.0
        if highcut >= nyquist:
            messagebox.showerror("Error",
                                 f"f0 + bandwidth/2 = {highcut:.3f} Hz exceeds Nyquist ({nyquist} Hz).")
            return

        # Extract crop and preprocess (DC removal + 1-4 Hz bandpass, same as analysis)
        start_idx, end_idx, name = self.crops[crop_idx]
        cropped_signal_raw = self.signal[start_idx:end_idx]
        cropped_time = self.time[start_idx:end_idx] - self.time[start_idx]
        preprocessed_signal = self.preprocess_signal(cropped_signal_raw, lowcut=0.5, highcut=4.0)

        # Design narrow Butterworth bandpass filter (SOS form)
        low = lowcut / nyquist
        high = highcut / nyquist
        order = 4
        sos = butter(order, [low, high], btype='band', output='sos')

        # Apply zero-phase filtering to the preprocessed signal
        try:
            min_padlen = 3 * (2 * len(sos))
            if len(preprocessed_signal) <= min_padlen:
                messagebox.showerror("Error",
                                     "Signal crop is too short for this filter configuration.\n"
                                     "Select a longer crop or increase bandwidth.")
                return
            filtered_signal = sosfiltfilt(sos, preprocessed_signal)
        except Exception as e:
            messagebox.showerror("Error", f"Filtering failed: {str(e)}")
            return

        # Show result in a new window
        filter_window = tk.Toplevel(self.root)
        filter_window.title(f"Narrow Band-Pass Filter - {name}")
        filter_window.geometry("1000x700")

        fig_filter, axes = plt.subplots(3, 1, figsize=(10, 8))

        # Plot 1: Preprocessed signal (DC removed + 1-4 Hz BP, same as analysis)
        axes[0].plot(cropped_time, preprocessed_signal, 'b-', linewidth=0.8)
        axes[0].set_title(f'{name} - Preprocessed (DC removed + 1-4 Hz BP)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Narrow band-pass filtered signal
        axes[1].plot(cropped_time, filtered_signal, 'r-', linewidth=1.0)
        axes[1].set_title(
            f'{name} - Narrow BPF  |  f0 = {f0:.2f} Hz, BW = {bw:.2f} Hz  '
            f'[{lowcut:.3f} - {highcut:.3f}] Hz  |  Zero-phase 4th-order Butterworth')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: FFT of filtered signal
        frequencies, fft_mag, fft_mag_db = self.compute_fft(filtered_signal)
        axes[2].plot(frequencies, fft_mag, 'g-', linewidth=0.8)
        axes[2].axvline(x=f0, color='r', linestyle='--', alpha=0.7, label=f'f0 = {f0:.2f} Hz')
        axes[2].axvline(x=lowcut, color='orange', linestyle=':', alpha=0.7,
                        label=f'Low = {lowcut:.3f} Hz')
        axes[2].axvline(x=highcut, color='orange', linestyle=':', alpha=0.7,
                        label=f'High = {highcut:.3f} Hz')
        axes[2].set_title(f'{name} - FFT of Filtered Signal')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Magnitude')
        axes[2].set_xlim(max(0, f0 - 2), f0 + 2)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')

        fig_filter.tight_layout()

        canvas = FigureCanvasTkAgg(fig_filter, master=filter_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, filter_window)
        toolbar.update()

    def find_peaks(self, data, threshold_ratio=0.1):
        """Find peaks in data."""
        threshold = np.max(data) * threshold_ratio
        peaks = []
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        
        # Sort by magnitude
        peaks.sort(key=lambda x: data[x], reverse=True)
        return peaks
    
    def create_summary_table(self, results_data):
        """Create a summary table of results."""
        table_frame = ttk.LabelFrame(self.results_inner_frame, text="Summary Table", padding="10")
        table_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Headers
        headers = ['Crop', 'Start (s)', 'End (s)', 'Duration (s)', 'Samples', 
                   'SNR (dB)', 'Signal RMS', 'Noise RMS', 'Peak Amp', 'Dynamic Range (dB)']
        
        for col, header in enumerate(headers):
            label = ttk.Label(table_frame, text=header, font=('Arial', 9, 'bold'), 
                             borderwidth=1, relief='solid', padding=5)
            label.grid(row=0, column=col, sticky='nsew')
        
        # Data rows
        for row, data in enumerate(results_data, start=1):
            values = [
                data['name'],
                f"{data['start_time']:.3f}",
                f"{data['end_time']:.3f}",
                f"{data['duration']:.3f}",
                str(data['samples']),
                f"{data['snr']['snr_db']:.2f}",
                f"{data['snr']['signal_rms']:.4f}",
                f"{data['snr']['noise_rms']:.4f}",
                f"{data['snr']['peak_amplitude']:.4f}",
                f"{data['snr']['dynamic_range_db']:.2f}"
            ]
            
            for col, value in enumerate(values):
                label = ttk.Label(table_frame, text=value, borderwidth=1, relief='solid', padding=5)
                label.grid(row=row, column=col, sticky='nsew')
        
        # Make columns expandable
        for col in range(len(headers)):
            table_frame.columnconfigure(col, weight=1)
            
    def save_results(self):
        """Save analysis results to files."""
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results to save. Please analyze crops first.")
            return
        
        # Ask for save location
        save_dir = filedialog.askdirectory(title="Select folder to save results")
        if not save_dir:
            return
        
        try:
            # Save figure
            self.results_figure.savefig(f"{save_dir}/analysis_results.png", dpi=150, bbox_inches='tight')
            
            # Save cropped signals as NPY files (both raw and preprocessed)
            for i, (start_idx, end_idx, name) in enumerate(self.crops):
                cropped_signal_raw = self.signal[start_idx:end_idx]
                cropped_signal_preprocessed = self.preprocess_signal(cropped_signal_raw, lowcut=1.0, highcut=4.0)
                
                # Save raw
                np.save(f"{save_dir}/{name.replace(' ', '_')}_raw.npy", cropped_signal_raw)
                # Save preprocessed
                np.save(f"{save_dir}/{name.replace(' ', '_')}_preprocessed.npy", cropped_signal_preprocessed)
            
            # Save results as JSON
            results_json = []
            for data in self.analysis_results:
                results_json.append({
                    'name': data['name'],
                    'start_time': data['start_time'],
                    'end_time': data['end_time'],
                    'duration': data['duration'],
                    'samples': data['samples'],
                    'preprocessing': 'DC removal (mean subtraction) + 1-4 Hz zero-phase bandpass filter (4th order Butterworth)',
                    'snr_db': data['snr']['snr_db'],
                    'signal_rms': data['snr']['signal_rms'],
                    'noise_rms': data['snr']['noise_rms'],
                    'peak_amplitude': data['snr']['peak_amplitude'],
                    'dynamic_range_db': data['snr']['dynamic_range_db']
                })
            
            with open(f"{save_dir}/analysis_results.json", 'w') as f:
                json.dump(results_json, f, indent=2)
            
            # Save crop info as CSV
            with open(f"{save_dir}/crop_info.csv", 'w') as f:
                f.write("Name,Start_Time,End_Time,Duration,Samples,SNR_dB,Signal_RMS,Noise_RMS,Peak_Amplitude,Dynamic_Range_dB\n")
                for data in self.analysis_results:
                    f.write(f"{data['name']},{data['start_time']:.6f},{data['end_time']:.6f},"
                           f"{data['duration']:.6f},{data['samples']},{data['snr']['snr_db']:.4f},"
                           f"{data['snr']['signal_rms']:.6f},{data['snr']['noise_rms']:.6f},"
                           f"{data['snr']['peak_amplitude']:.6f},{data['snr']['dynamic_range_db']:.4f}\n")
            
            messagebox.showinfo("Success", f"Results saved to {save_dir}\n\nFiles saved:\n- analysis_results.png\n- *_raw.npy (raw signals)\n- *_preprocessed.npy (filtered signals)\n- analysis_results.json\n- crop_info.csv")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            
    def load_signal(self):
        """Load a signal from file."""
        file_path = filedialog.askopenfilename(
            title="Select signal file",
            filetypes=[("NumPy files", "*.npy"), ("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.npy'):
                self.signal = np.load(file_path)
            elif file_path.endswith('.csv') or file_path.endswith('.txt'):
                self.signal = np.loadtxt(file_path, delimiter=',')
                if self.signal.ndim > 1:
                    # Take first column or flatten
                    self.signal = self.signal.flatten() if self.signal.shape[1] == 1 else self.signal[:, 0]
            else:
                # Try to load as text
                self.signal = np.loadtxt(file_path)
            
            # Update time array
            self.time = np.arange(len(self.signal)) / self.sample_rate
            
            # Clear existing crops
            self.crops.clear()
            self.crops_listbox.delete(0, tk.END)
            self.crop_counter = 1
            
            # Replot
            self.plot_signal()
            
            messagebox.showinfo("Success", f"Signal loaded: {len(self.signal)} samples")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load signal: {str(e)}")
            
    def update_sample_rate(self, event=None):
        """Update the sample rate."""
        try:
            new_rate = float(self.sample_rate_var.get())
            if new_rate <= 0:
                raise ValueError("Sample rate must be positive")
            
            self.sample_rate = new_rate
            self.time = np.arange(len(self.signal)) / self.sample_rate
            
            # Update crops listbox
            self.crops_listbox.delete(0, tk.END)
            for start_idx, end_idx, name in self.crops:
                start_time = start_idx / self.sample_rate
                end_time = end_idx / self.sample_rate
                self.crops_listbox.insert(tk.END, f"{name}: {start_time:.3f}s - {end_time:.3f}s")
            
            # Replot
            self.plot_signal()
            
            messagebox.showinfo("Success", f"Sample rate updated to {new_rate} Hz")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid sample rate: {str(e)}")

    def calculate_window_quality(self, signal, cardiac_band=(0.8, 2.5)):
        """
        Calculate quality score for a signal window based on multiple metrics.
        
        Metrics:
        - Spectral Concentration: Energy concentrated around dominant peak (MOST IMPORTANT)
        - Peak Count Penalty: Fewer significant peaks = cleaner signal
        - Peak Prominence: How much the dominant peak stands out
        - Harmonic Detection: Presence of harmonic at 2*f0 indicates biological signal
        
        Returns:
            dict with metrics and quality_score (0-1 scale, higher = better)
        """
        n = len(signal)
        if n < 30:
            return {'valid': False, 'reason': 'too_short'}
        
        # Preprocess: DC removal only (no bandpass to preserve harmonics for detection)
        signal_no_dc = signal - np.mean(signal)
        
        # 0. Artifact Rejection (Time Domain)
        signal_median = np.median(signal_no_dc)
        abs_deviation = np.abs(signal_no_dc - signal_median)
        mad = np.median(abs_deviation)
        
        if mad < 1e-10:
            mad = 1e-10
            
        max_deviation = np.max(abs_deviation)
        artifact_ratio = max_deviation / mad
        
        if artifact_ratio > 6.0:
            return {'valid': False, 'reason': f'artifact_detected_ratio_{artifact_ratio:.1f}'}
        
        # Compute FFT with Hanning window
        window = np.hanning(n)
        windowed_signal = signal_no_dc * window
        fft_result = np.fft.fft(windowed_signal)
        fft_magnitude = np.abs(fft_result[:n // 2])
        frequencies = np.fft.fftfreq(n, 1 / self.sample_rate)[:n // 2]
        
        # Find cardiac band indices
        cardiac_mask = (frequencies >= cardiac_band[0]) & (frequencies <= cardiac_band[1])
        
        if not np.any(cardiac_mask):
            return {'valid': False, 'reason': 'no_cardiac_band'}
        
        cardiac_freqs = frequencies[cardiac_mask]
        cardiac_magnitude = fft_magnitude[cardiac_mask]
        cardiac_power = cardiac_magnitude ** 2
        
        if np.sum(cardiac_power) < 1e-10:
            return {'valid': False, 'reason': 'no_power'}
        
        # 1. Find dominant frequency (highest peak in cardiac band)
        peak_idx = np.argmax(cardiac_magnitude)
        dominant_freq = cardiac_freqs[peak_idx]
        peak_power = cardiac_power[peak_idx]
        peak_magnitude = cardiac_magnitude[peak_idx]
        
        # ============================================================
        # NEW METRIC 1: Spectral Concentration (Most Important!)
        # ============================================================
        # Measures how much energy is concentrated around the dominant peak
        # Clean signal: most energy in narrow band around peak
        # Dirty signal: energy spread across many frequencies
        
        # Define narrow band around dominant frequency (±0.2 Hz)
        narrow_band_width = 0.2  # Hz
        narrow_band_mask = (cardiac_freqs >= dominant_freq - narrow_band_width) & \
                          (cardiac_freqs <= dominant_freq + narrow_band_width)
        
        if np.any(narrow_band_mask):
            narrow_band_power = np.sum(cardiac_power[narrow_band_mask])
            total_cardiac_power = np.sum(cardiac_power)
            spectral_concentration = narrow_band_power / (total_cardiac_power + 1e-10)
        else:
            spectral_concentration = 0.0
        
        # Normalize: concentration of 0.3 = minimum (0), 1.0 = maximum (1)
        spectral_concentration_normalized = np.clip((spectral_concentration - 0.3) / 0.7, 0, 1)
        
        # ============================================================
        # STRICT METRIC: Non-Harmonic Peak Count (CRITICAL!)
        # ============================================================
        # If there's more than 1 significant peak that is NOT a harmonic,
        # the signal is considered dirty and should be heavily penalized.
        
        # A peak is "significant" if it's > 15% of the dominant peak (stricter threshold)
        significance_threshold = peak_magnitude * 0.15
        
        # Detect harmonic frequency (2*f0) - this is allowed
        harmonic_freq = 2 * dominant_freq
        harmonic_tolerance = 0.15  # Hz tolerance for harmonic detection
        
        # Find all significant peaks in cardiac band
        significant_peak_freqs = []
        for i in range(1, len(cardiac_magnitude) - 1):
            if (cardiac_magnitude[i] > cardiac_magnitude[i-1] and 
                cardiac_magnitude[i] > cardiac_magnitude[i+1] and
                cardiac_magnitude[i] > significance_threshold):
                significant_peak_freqs.append(cardiac_freqs[i])
        
        # Count non-harmonic peaks (exclude dominant peak and its harmonic)
        non_harmonic_peaks = 0
        for peak_freq in significant_peak_freqs:
            # Skip if this is the dominant peak
            if abs(peak_freq - dominant_freq) < 0.1:
                continue
            # Skip if this is the harmonic (2*f0)
            if abs(peak_freq - harmonic_freq) < harmonic_tolerance:
                continue
            # This is an extra, non-harmonic peak - BAD!
            non_harmonic_peaks += 1
        
        # STRICT PENALTY: Any non-harmonic extra peak = signal is dirty
        # 0 extra peaks = 1.0 (clean), 1 extra peak = 0.2 (bad), 2+ extra peaks = 0.0 (very bad)
        if non_harmonic_peaks == 0:
            peak_count_score = 1.0
        elif non_harmonic_peaks == 1:
            peak_count_score = 0.2
        else:
            peak_count_score = 0.0
        
        # ============================================================
        # Existing Metrics (with reduced weight)
        # ============================================================
        
        # 3. Peak-to-Median Ratio in cardiac band
        cardiac_median = np.median(cardiac_magnitude)
        if cardiac_median < 1e-10:
            cardiac_median = 1e-10
        peak_to_median = peak_magnitude / cardiac_median
        peak_to_median_normalized = np.clip((peak_to_median - 1) / 15, 0, 1)
        
        # 4. Peak Prominence (from full spectrum noise floor)
        noise_freq_mask = frequencies > 0.5
        if np.any(noise_freq_mask):
            noise_magnitudes = fft_magnitude[noise_freq_mask]
            noise_floor = np.median(noise_magnitudes[noise_magnitudes > 0])
        else:
            noise_floor = np.median(fft_magnitude[fft_magnitude > 0])
        if noise_floor < 1e-10:
            noise_floor = 1e-10
        prominence_ratio = peak_magnitude / noise_floor
        prominence_normalized = np.clip((prominence_ratio - 1) / 15, 0, 1)
        
        # 5. SNR in cardiac band
        noise_power = total_cardiac_power - peak_power
        if noise_power < 1e-10:
            snr_cardiac = 30.0
        else:
            snr_cardiac = 10 * np.log10(peak_power / noise_power)
        snr_normalized = np.clip((snr_cardiac + 10) / 40, 0, 1)
        
        # 6. Harmonic detection (bonus)
        harmonic_freq = 2 * dominant_freq
        harmonic_bonus = 0.0
        harmonic_detected = False
        
        if harmonic_freq <= 5.0:
            harmonic_mask = (frequencies >= harmonic_freq * 0.9) & (frequencies <= harmonic_freq * 1.1)
            if np.any(harmonic_mask):
                harmonic_magnitude = np.max(fft_magnitude[harmonic_mask])
                harmonic_power = harmonic_magnitude ** 2
                if (harmonic_power > peak_power * 0.001 and 
                    harmonic_power < peak_power and
                    harmonic_magnitude > noise_floor * 2):
                    harmonic_bonus = 1.0
                    harmonic_detected = True
        
        # ============================================================
        # QUALITY SCORE FORMULA (Strict non-harmonic peak penalty)
        # ============================================================
        # Peak Count Score (35%) - MOST IMPORTANT: Extra non-harmonic peaks = dirty signal
        # Spectral Concentration (30%) - Clean signals concentrate energy
        # Peak-to-Median (15%) - How dominant the peak is
        # Harmonic Bonus (10%) - Biological signal indicator
        # SNR (10%) - Signal to noise ratio
        
        quality_score = (
            peak_count_score * 0.35 +
            spectral_concentration_normalized * 0.30 +
            peak_to_median_normalized * 0.15 +
            harmonic_bonus * 0.10 +
            snr_normalized * 0.10
        )
        
        return {
            'valid': True,
            'spectral_concentration': float(spectral_concentration),
            'spectral_concentration_normalized': float(spectral_concentration_normalized),
            'non_harmonic_peaks': int(non_harmonic_peaks),
            'peak_count_score': float(peak_count_score),
            'peak_to_median': float(peak_to_median),
            'peak_to_median_normalized': float(peak_to_median_normalized),
            'prominence_ratio': float(prominence_ratio),
            'prominence_normalized': float(prominence_normalized),
            'snr_cardiac_db': float(snr_cardiac),
            'snr_normalized': float(snr_normalized),
            'dominant_freq': float(dominant_freq),
            'dominant_freq_bpm': float(dominant_freq * 60),
            'harmonic_detected': harmonic_detected,
            'harmonic_bonus': float(harmonic_bonus),
            'quality_score': float(quality_score)
        }

    def find_best_opa_windows(self, window_duration=5.0, step_samples=1, top_n=5, cardiac_band=(0.8, 2.5), apply_bandpass=True):
        """
        Find the best OPA signal windows using sliding window analysis.
        
        Args:
            window_duration: Window size in seconds
            step_samples: Step size in frames/samples
            top_n: Return top N best windows
            cardiac_band: Frequency range for cardiac signal (Hz)
            apply_bandpass: If True, apply 1-4 Hz bandpass. If False, only DC removal.
        
        Returns:
            List of dicts with start_idx, end_idx, start_time, end_time, and quality metrics
        """
        window_samples = int(window_duration * self.sample_rate)
        
        if window_samples > len(self.signal):
            return []
        
        results = []
        
        # Slide window across signal
        for start_idx in range(0, len(self.signal) - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_signal = self.signal[start_idx:end_idx]
            
            # Preprocess signal based on apply_bandpass setting
            if apply_bandpass:
                preprocessed = self.preprocess_signal(window_signal, lowcut=1.0, highcut=4.0)
            else:
                preprocessed = window_signal - np.mean(window_signal)  # Just DC removal
            
            # Calculate quality metrics on preprocessed signal
            quality = self.calculate_window_quality(preprocessed, cardiac_band)
            
            if quality.get('valid', False):
                results.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': start_idx / self.sample_rate,
                    'end_time': end_idx / self.sample_rate,
                    **quality
                })
        
        # Sort by quality score (descending)
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Remove overlapping windows (keep higher quality one)
        filtered_results = []
        for result in results:
            is_overlapping = False
            for existing in filtered_results:
                # Check for significant overlap (>50%)
                overlap_start = max(result['start_idx'], existing['start_idx'])
                overlap_end = min(result['end_idx'], existing['end_idx'])
                overlap_size = max(0, overlap_end - overlap_start)
                
                if overlap_size > window_samples * 0.5:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_results.append(result)
                if len(filtered_results) >= top_n:
                    break
        
        return filtered_results

    def auto_find_best_opa(self):
        """
        Automatically find and analyze the best OPA signal windows.
        """
        # Get parameters from UI
        try:
            window_duration = float(self.window_length_var.get())
            if window_duration <= 0:
                raise ValueError("Window length must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid window length: {str(e)}")
            return
        
        try:
            top_k = int(self.top_k_var.get())
            if top_k <= 0:
                raise ValueError("Top K must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid Top K value: {str(e)}")
            return
        
        try:
            stride_frames = int(self.stride_var.get())
            if stride_frames <= 0:
                raise ValueError("Stride must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid Stride value: {str(e)}")
            return
        
        # Get bandpass filter setting from checkbox
        apply_bandpass = self.apply_bandpass_var.get()
        
        # Find best windows
        self.auto_find_btn.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            best_windows = self.find_best_opa_windows(
                window_duration=window_duration,
                step_samples=stride_frames,
                top_n=top_k,
                apply_bandpass=apply_bandpass
            )
        finally:
            self.auto_find_btn.config(state=tk.NORMAL)
        
        if not best_windows:
            messagebox.showwarning("Warning", "No valid OPA windows found in the signal.")
            return
        
        # Clear existing crops and add the best windows as crops
        self.crops.clear()
        self.crops_listbox.delete(0, tk.END)
        self.crop_counter = 1
        
        for i, window in enumerate(best_windows):
            crop_name = f"Auto {i+1} (Q={window['quality_score']:.2f})"
            self.crops.append((window['start_idx'], window['end_idx'], crop_name))
            
            start_time = window['start_time']
            end_time = window['end_time']
            self.crops_listbox.insert(tk.END, f"{crop_name}: {start_time:.2f}s - {end_time:.2f}s")
            self.crop_counter += 1
        
        # Redraw patches on signal plot
        self.redraw_crop_patches()
        
        # Automatically analyze the crops (same as clicking "Analyze Crops")
        self.analyze_crops()
        
        # Show summary message
        summary_msg = f"Found {len(best_windows)} best OPA windows:\n\n"
        for i, w in enumerate(best_windows):
            summary_msg += (f"{i+1}. {w['start_time']:.1f}s - {w['end_time']:.1f}s\n"
                          f"   Quality: {w['quality_score']:.3f}, "
                          f"HR: {w['dominant_freq_bpm']:.0f} BPM, "
                          f"SNR: {w['snr_cardiac_db']:.1f} dB\n")
        
        messagebox.showinfo("Auto-Find Results", summary_msg)

    def analyze_folder(self):
        """
        Analyze all patient folders in a selected directory.
        Expects structure: parent_folder/<patient>/<subfolder>/<signal>.npy
        """
        # Get folder path
        folder_path = filedialog.askdirectory(title="Select Parent Folder")
        if not folder_path:
            return
        
        # Get parameters from UI
        signal_name = self.batch_signal_var.get().strip()
        if not signal_name:
            signal_name = "opa_lateral_percent"
        
        subfolder = self.batch_subfolder_var.get().strip()
        if not subfolder:
            subfolder = "left_eye"
        
        try:
            top_k = int(self.batch_top_k_var.get())
            if top_k <= 0:
                raise ValueError("Top K must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid Top K value: {str(e)}")
            return
        
        # Scan for patient folders
        patient_folders = []
        for item in sorted(os.listdir(folder_path)):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                signal_file = os.path.join(item_path, subfolder, f"{signal_name}.npy")
                if os.path.exists(signal_file):
                    patient_folders.append({
                        'name': item,
                        'signal_path': signal_file
                    })
        
        if not patient_folders:
            messagebox.showwarning("Warning", 
                f"No valid patient folders found.\n\n"
                f"Expected structure:\n"
                f"  {folder_path}/\n"
                f"    <patient>/\n"
                f"      {subfolder}/\n"
                f"        {signal_name}.npy")
            return
        
        # Disable button during processing
        self.analyze_folder_btn.config(state=tk.DISABLED)
        self.root.update()
        
        # Analyze each patient
        all_results = []
        failed_patients = []
        
        for patient in patient_folders:
            try:
                # Load signal
                patient_signal = np.load(patient['signal_path'])
                
                # Temporarily set signal for analysis
                original_signal = self.signal
                original_time = self.time
                self.signal = patient_signal
                self.time = np.arange(len(patient_signal)) / self.sample_rate
                
                # Get window duration from UI
                try:
                    window_duration = float(self.window_length_var.get())
                except:
                    window_duration = 5.0
                
                try:
                    stride_frames = int(self.stride_var.get())
                except:
                    stride_frames = 1
                
                # Get bandpass filter setting
                apply_bandpass = self.apply_bandpass_var.get()
                
                # Find best windows
                best_windows = self.find_best_opa_windows(
                    window_duration=window_duration,
                    step_samples=stride_frames,
                    top_n=top_k,
                    apply_bandpass=apply_bandpass
                )
                
                # Restore original signal
                self.signal = original_signal
                self.time = original_time
                
                all_results.append({
                    'patient': patient['name'],
                    'signal': patient_signal,
                    'windows': best_windows
                })
                
            except Exception as e:
                failed_patients.append(f"{patient['name']}: {str(e)}")
        
        self.analyze_folder_btn.config(state=tk.NORMAL)
        
        if not all_results:
            messagebox.showerror("Error", "Failed to analyze any patients.\n\n" + "\n".join(failed_patients))
            return
        
        # Create grid visualization
        self._create_batch_grid(all_results, top_k, signal_name)
        
        # Show summary
        summary = f"Analyzed {len(all_results)} patients"
        if failed_patients:
            summary += f"\n\nFailed ({len(failed_patients)}):\n" + "\n".join(failed_patients[:5])
        messagebox.showinfo("Batch Analysis Complete", summary)
    
    def _create_batch_grid(self, all_results, top_k, signal_name):
        """Create a grid figure showing FFTs for all patients - similar to analyze_crops style."""
        num_patients = len(all_results)
        
        # Get bandpass filter setting from checkbox
        apply_bandpass = self.apply_bandpass_var.get()
        
        # Create new window for results
        grid_window = tk.Toplevel(self.root)
        grid_window.title(f"Batch Analysis: {signal_name}")
        grid_window.state('zoomed')  # Maximize window
        
        # Create scrollable frame
        main_frame = ttk.Frame(grid_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar for vertical scrolling
        canvas = tk.Canvas(main_frame)
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create inner frame for content
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
        
        # Create figure with appropriate size
        fig_width = 4 * top_k  # 4 inches per column
        fig_height = 3 * num_patients  # 3 inches per row
        fig, axes = plt.subplots(num_patients, top_k, figsize=(fig_width, fig_height))
        
        # Handle single row/column cases
        if num_patients == 1 and top_k == 1:
            axes = np.array([[axes]])
        elif num_patients == 1:
            axes = axes.reshape(1, -1)
        elif top_k == 1:
            axes = axes.reshape(-1, 1)
        
        # Column headers
        rank_labels = ["Best", "2nd Best", "3rd Best", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        
        for row_idx, result in enumerate(all_results):
            patient_name = result['patient']
            windows = result['windows']
            patient_signal = result['signal']
            
            # Shorten patient name
            short_name = patient_name.replace('one_eye_new_', '').replace('one_eye_', '')
            
            for col_idx in range(top_k):
                ax = axes[row_idx, col_idx]
                
                if col_idx < len(windows):
                    window = windows[col_idx]
                    
                    # Extract and preprocess window signal
                    start_idx = window['start_idx']
                    end_idx = window['end_idx']
                    window_signal = patient_signal[start_idx:end_idx]
                    
                    # Apply preprocessing based on checkbox setting
                    if apply_bandpass:
                        preprocessed = self.preprocess_signal(window_signal, lowcut=0.5, highcut=4.0)
                    else:
                        preprocessed = window_signal - np.mean(window_signal)  # Just DC removal
                    
                    # Compute FFT
                    frequencies, fft_magnitude, _ = self.compute_fft(preprocessed)
                    
                    # Plot FFT (same style as analyze_crops)
                    ax.plot(frequencies, fft_magnitude, 'r-', linewidth=1.0)
                    ax.set_xlim(0, 10)
                    ax.grid(True, alpha=0.3)
                    
                    # Find and annotate dominant peaks (same as analyze_crops)
                    peak_indices = self.find_peaks(fft_magnitude)
                    for peak_idx in peak_indices[:3]:  # Show top 3 peaks
                        if fft_magnitude[peak_idx] > np.max(fft_magnitude) * 0.1:
                            ax.annotate(f'{frequencies[peak_idx]:.1f} Hz',
                                       xy=(frequencies[peak_idx], fft_magnitude[peak_idx]),
                                       fontsize=8, ha='center', va='bottom')
                    
                    # Quality and HR in title
                    q = window['quality_score']
                    hr = window['dominant_freq_bpm']
                    
                    # Title format: Patient name + Quality + HR
                    if col_idx == 0:
                        ax.set_title(f'{short_name}\nQ={q:.2f}, {hr:.0f} BPM', fontsize=9, fontweight='bold')
                    else:
                        ax.set_title(f'Q={q:.2f}, {hr:.0f} BPM', fontsize=9)
                    
                    # Axis labels
                    ax.set_xlabel('Frequency (Hz)', fontsize=8)
                    ax.set_ylabel('Magnitude', fontsize=8)
                    ax.tick_params(axis='both', labelsize=7)
                    
                else:
                    # No data for this column
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12, 
                           color='gray', style='italic', transform=ax.transAxes)
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 1)
                    ax.set_facecolor('#f8f8f8')
                    ax.set_xlabel('Frequency (Hz)', fontsize=8)
                    ax.set_ylabel('Magnitude', fontsize=8)
                    
                    if col_idx == 0:
                        ax.set_title(f'{short_name}\nN/A', fontsize=9, fontweight='bold')
                    else:
                        ax.set_title('N/A', fontsize=9)
        
        # Add column headers as figure text
        for col_idx in range(top_k):
            header = rank_labels[col_idx] if col_idx < len(rank_labels) else f"{col_idx+1}th"
            x_pos = (col_idx + 0.5) / top_k
            fig.text(x_pos, 0.995, header, ha='center', va='top', fontsize=10, fontweight='bold')
        
        fig.suptitle(f'Batch OPA Analysis - {signal_name} (Top {top_k} Windows per Patient)', 
                     fontsize=12, fontweight='bold', y=1.01)
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.96, hspace=0.35, wspace=0.25)
        
        # Embed figure in tkinter
        fig_canvas = FigureCanvasTkAgg(fig, master=inner_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure scroll region
        inner_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Add toolbar in separate frame at bottom
        toolbar_frame = ttk.Frame(grid_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(fig_canvas, toolbar_frame)
        toolbar.update()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    
    # Try to set a theme
    try:
        style = ttk.Style()
        style.theme_use('clam')
    except:
        pass
    
    # Create app with 60 Hz sample rate (video FPS)
    app = SignalCropperApp(root, sample_rate=60)
    
    root.mainloop()


if __name__ == "__main__":
    main()

