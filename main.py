import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class CoinAnalysisGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Coin Analysis Parameters")
        
        # Initialize instance variables
        self.current_file = None
        self.img = None
        self.output = None
        
        # สร้างเฟรมสำหรับควบคุม
        self.control_frame = ttk.LabelFrame(root, text="Parameter Controls", padding="10")
        self.control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # สร้างตัวแปรสำหรับเก็บค่าพารามิเตอร์
        self.min_dist = tk.IntVar(value=145)
        self.param1 = tk.IntVar(value=15)
        self.param2 = tk.IntVar(value=30)
        self.min_radius = tk.IntVar(value=7)
        self.max_radius = tk.IntVar(value=155)
        self.blur_kernel = tk.IntVar(value=7)
        
        # เพิ่มตัวแปรสำหรับเก็บข้อมูลวงกลมที่ตรวจพบ
        self.detected_regions = []
        self.reference_coin = tk.StringVar(value="1")
        
        # สร้าง widgets สำหรับควบคุมค่าต่างๆ
        self.create_parameter_controls()
        
        # สร้างส่วนเลือกเหรียญอ้างอิง
        self.create_reference_selection()
        
        # สร้างปุ่มสำหรับประมวลผล
        self.create_action_buttons()
        
        # สร้างพื้นที่แสดงภาพ
        self.image_label = ttk.Label(root)
        self.image_label.grid(row=0, column=1, padx=10, pady=5, rowspan=3)
        
        self.output_image = None
        self.current_image = None
        
    def create_reference_selection(self):
        ref_frame = ttk.LabelFrame(self.root, text="Reference Coin Selection", padding="10")
        ref_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        
        ttk.Label(ref_frame, text="Select reference coin:").pack(padx=5, pady=2)
        self.coin_dropdown = ttk.Combobox(ref_frame, textvariable=self.reference_coin, state='readonly')
        self.coin_dropdown.pack(padx=5, pady=2)
        
        ttk.Button(ref_frame, text="Compare", command=self.update_comparison).pack(padx=5, pady=5)

    def create_parameter_controls(self):
        parameters = [
            ("Minimum Distance", self.min_dist, 50, 300),
            ("Parameter 1", self.param1, 1, 100),
            ("Parameter 2", self.param2, 1, 100),
            ("Minimum Radius", self.min_radius, 1, 100),
            ("Maximum Radius", self.max_radius, 50, 300),
            ("Blur Kernel Size", self.blur_kernel, 3, 21, 2)
        ]
        
        for i, (label, var, min_val, max_val, *args) in enumerate(parameters):
            ttk.Label(self.control_frame, text=label).grid(row=i, column=0, padx=5, pady=2)
            step = args[0] if args else 1
            slider = ttk.Scale(
                self.control_frame,
                from_=min_val,
                to=max_val,
                variable=var,
                orient="horizontal"
            )
            slider.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            ttk.Label(self.control_frame, textvariable=var).grid(row=i, column=2, padx=5, pady=2)

    def create_action_buttons(self):
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Process", command=self.process_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Reset Parameters", command=self.reset_parameters).grid(row=0, column=2, padx=5)

    def reset_parameters(self):
        self.min_dist.set(145)
        self.param1.set(15)
        self.param2.set(30)
        self.min_radius.set(7)
        self.max_radius.set(155)
        self.blur_kernel.set(7)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            # Convert file path to absolute path with proper encoding
            abs_path = os.path.abspath(file_path)
            try:
                # Try reading with numpy first (handles Unicode paths better)
                img_array = np.fromfile(abs_path, np.uint8)
                self.img = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                if self.img is not None:
                    self.current_file = abs_path
                    self.current_image = self.img.copy()
                    self.output = self.img.copy()
                    self.show_image(self.img)
                    self.detected_circles = None
                    self.coin_dropdown['values'] = []
                    self.reference_coin.set("")
                else:
                    raise ValueError("Could not decode image")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")


    def extract_circle_region(self, image, x, y, r):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv.circle(mask, (x, y), r, 255, -1)
        masked_image = cv.bitwise_and(image, image, mask=mask)
        return masked_image, mask

    def calculate_histogram(self, image, mask):
        """
        Calculate color histogram for a masked image region
    
        Parameters:
        -----------
        image : ndarray
            Input image in BGR format
        mask : ndarray
            Binary mask of same size as image
        """
        # Convert mask to boolean array
        mask_bool = mask > 0
    
        # Extract masked pixels
        masked_pixels = image[mask_bool]
    
        if masked_pixels.size == 0:
            return np.zeros(8 * 8 * 8)
        
        # Define histogram parameters
        bins = (8, 8, 8)
        ranges = ((0, 256), (0, 256), (0, 256))
    
        # Calculate bin widths for each channel
        bin_widths = [
            (ranges[i][1] - ranges[i][0]) / bins[i] 
            for i in range(3)
        ]
    
        # Initialize histogram array
        hist = np.zeros(bins)
    
        # Calculate bin indices for each pixel
        bin_indices = [
            np.clip(
                np.floor((masked_pixels[:, i] - ranges[i][0]) / bin_widths[i]).astype(int),
                0, bins[i] - 1
            )
            for i in range(3)
        ]
    
        # Count occurrences in each bin
        for pixel_bins in zip(*bin_indices):
            hist[pixel_bins] += 1
    
        # Normalize histogram
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
    
        return hist.flatten()

    def calculate_histogram_similarity(self, hist1, hist2):
        """
        Calculate correlation between two histograms
        """
        # Calculate correlation coefficient
        mean1, mean2 = np.mean(hist1), np.mean(hist2)
        std1, std2 = np.std(hist1), np.std(hist2)
    
        if std1 == 0 or std2 == 0:
            return 0.0
    
        correlation = np.mean(
            ((hist1 - mean1) * (hist2 - mean2)) / (std1 * std2)
        )
    
        # Convert to percentage and ensure it's in valid range
        return max(min(correlation * 100, 100), 0)

    def custom_hough_circles(self, gray_image, min_dist, param1, param2, min_radius, max_radius):
        """
        Custom implementation of Hough Circle Transform
        
        Parameters:
        -----------
        gray_image : ndarray
            Input grayscale image
        min_dist : int
            Minimum distance between circle centers
        param1 : int
            Gradient value used for edge detection
        param2 : int
            Accumulator threshold for circle detection
        min_radius : int
            Minimum circle radius
        max_radius : int
            Maximum circle radius
            
        Returns:
        --------
        circles : ndarray
            Detected circles in format [[x, y, radius], ...]
        """
        height, width = gray_image.shape
        
        # Calculate image gradients using Sobel
        dx = np.gradient(gray_image.astype(float), axis=1)
        dy = np.gradient(gray_image.astype(float), axis=0)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        
        # Create accumulator array for circle centers
        accumulator = np.zeros((height, width))
        
        # Edge detection threshold
        edge_threshold = param1
        
        # Get edge points
        edge_points = np.where(magnitude > edge_threshold)
        edge_points = list(zip(edge_points[0], edge_points[1]))
        
        # For each edge point, vote in accumulator space
        for y, x in edge_points:
            # Consider potential circles through this edge point
            for r in range(min_radius, max_radius + 1):
                # Use gradient direction to vote for circle center
                theta = direction[y, x]
                # Calculate potential center coordinates
                a = int(x - r * np.cos(theta))
                b = int(y - r * np.sin(theta))
                
                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a] += 1
        
        # Find peaks in accumulator array
        circles = []
        accumulator_threshold = param2
        
        # Non-maximum suppression
        for y in range(height):
            for x in range(width):
                if accumulator[y, x] > accumulator_threshold:
                    # Check if it's a local maximum
                    local_max = True
                    window_size = min_dist // 2
                    
                    for dy in range(-window_size, window_size + 1):
                        for dx in range(-window_size, window_size + 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < height and 0 <= nx < width and 
                                (ny != y or nx != x) and 
                                accumulator[ny, nx] >= accumulator[y, x]):
                                local_max = False
                                break
                        if not local_max:
                            break
                    
                    if local_max:
                        # Find best radius for this center
                        best_radius = min_radius
                        max_votes = 0
                        
                        for r in range(min_radius, max_radius + 1):
                            votes = 0
                            # Count votes along circle perimeter
                            for theta in np.linspace(0, 2*np.pi, 36):
                                px = int(x + r * np.cos(theta))
                                py = int(y + r * np.sin(theta))
                                if (0 <= px < width and 0 <= py < height and 
                                    magnitude[py, px] > edge_threshold):
                                    votes += 1
                            
                            if votes > max_votes:
                                max_votes = votes
                                best_radius = r
                        
                        circles.append([x, y, best_radius])
        
        # Convert to numpy array and reshape to match cv2.HoughCircles output format
        if circles:
            circles = np.array(circles)[np.newaxis, :]
            return circles
        return None
    
    def custom_median_blur(self, image, kernel_size):
        """
        Applies custom median blur on the image.
        
        Parameters:
        -----------
        image : ndarray
            Input image (BGR format)
        kernel_size : int
            Size of the kernel window (must be odd)
            
        Returns:
        --------
        blurred_image : ndarray
            Blurred output image
        """
        height, width = image.shape[:2]
        pad = kernel_size // 2
        blurred_image = np.copy(image)
        
        # Pad the image to handle border issues
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

        for i in range(height):
            for j in range(width):
                # Define the region (window) around the current pixel
                window = padded_image[i:i + kernel_size, j:j + kernel_size]
                # Apply median on the window for each color channel
                for c in range(3):  # For each channel (RGB)
                    blurred_image[i, j, c] = np.median(window[:, :, c])
        
        return blurred_image
    
    def show_image(self, cv_image):
        if cv_image is not None:
            cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
            height, width = cv_image.shape[:2]
            
            max_size = 800
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                cv_image = cv.resize(cv_image, (new_width, new_height))
            
            pil_image = Image.fromarray(cv_image)
            photo = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def update_comparison(self):
        if not self.detected_regions:
            return
            
        output = self.output_image.copy()
        ref_idx = int(self.reference_coin.get()) - 1
        ref_region = self.detected_regions[ref_idx]
        
        # เน้นเหรียญอ้างอิง
        x, y, r = ref_region['position']
        cv.circle(output, (x, y), r, (0, 255, 0), 3)  # สีเขียวสำหรับเหรียญอ้างอิง
        
        # คำนวณและแสดงความเหมือนกับเหรียญอื่น
        for i, region in enumerate(self.detected_regions):
            if i != ref_idx:
                x, y, r = region['position']
                similarity = self.calculate_histogram_similarity(
                    ref_region['histogram'],
                    region['histogram']
                )
                
                # เลือกสีตามค่าความเหมือน
                if similarity >= 90:
                    color = (0, 255, 0)  # เขียว: เหมือนมาก
                elif similarity >= 70:
                    color = (255, 165, 0)  # ส้ม: เหมือนปานกลาง
                else:
                    color = (255, 0, 0)  # แดง: เหมือนน้อย
                
                cv.circle(output, (x, y), r, color, 3)
                # แสดงเปอร์เซ็นต์ความเหมือนเหนือเหรียญ
                cv.putText(output, f"{similarity:.1f}%", 
                          (x + 70, y - r + 70),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        self.show_image(output)

    def process_image(self):
        if self.current_image is None:
            return

        output = self.current_image.copy()
        # Convert to grayscale
        gray = cv.cvtColor(self.current_image, cv.COLOR_BGR2GRAY)
        
        # Apply custom median blur instead of cv.medianBlur
        blurred = self.custom_median_blur(self.current_image, self.blur_kernel.get())
        gray_blurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

        # Use custom Hough circles implementation
        circles = self.custom_hough_circles(
            gray_blurred,
            self.min_dist.get(),
            self.param1.get(),
            self.param2.get(),
            self.min_radius.get(),
            self.max_radius.get()
        )

        if circles is not None:
            detected_circles = np.uint16(np.around(circles))
            self.detected_regions = []
            
            for i, (x, y, r) in enumerate(detected_circles[0, :]):
                region, mask = self.extract_circle_region(self.current_image, x, y, r)
                hist = self.calculate_histogram(region, mask)
                self.detected_regions.append({
                    'position': (x, y, r),
                    'histogram': hist
                })
                
                cv.circle(output, (x, y), r, (0, 0, 0), 3)
                cv.circle(output, (x, y), 2, (0, 255, 255), 3)
                cv.putText(output, f"#{i+1}", (x - 20, y + r + 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            coin_numbers = [str(i+1) for i in range(len(self.detected_regions))]
            self.coin_dropdown['values'] = coin_numbers
            if not self.reference_coin.get() in coin_numbers:
                self.reference_coin.set(coin_numbers[0])

        self.show_image(output)
        self.output_image = output

def main():
    root = tk.Tk()
    app = CoinAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()