import os
from fpdf import FPDF

class VerifixiaReportPDF(FPDF):
    def header(self):
        # Draw a premium dark blue top banner background
        self.set_fill_color(26, 54, 93)  # Hex: #1A365D
        self.rect(0, 0, 210, 40, "F")
        
        # Title text
        self.set_xy(10, 10)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "VERIFIXIA AI PLATFORM", ln=True, align="L")
        
        # Subtitle text
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(200, 220, 240)
        self.cell(0, 5, "Technical Specifications & Deep Learning Model Architecture Report", ln=True, align="L")
        
        # Spacer
        self.set_y(45)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(113, 128, 150)  # Neutral gray
        
        # Draw footer line
        self.line(10, 280, 200, 280)
        
        # Footer text
        self.cell(0, 10, f"Verifixia Model Report | Page {self.page_no()}/{{nb}}", align="L")
        self.set_x(-60)
        self.cell(0, 10, "Confidential - Internal Research & Development", align="R")

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(43, 108, 176)  # Blue color
        self.cell(0, 8, f"{num}. {title}", ln=True, align="L")
        # Line below title
        self.set_draw_color(43, 108, 176)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(45, 55, 72)
        self.cell(0, 6, title, ln=True, align="L")
        self.ln(2)

    def body_text(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(74, 85, 104)
        if indent > 0:
            self.set_x(self.get_x() + indent)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def spec_table(self, specs):
        # Table of key-value parameters
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(247, 250, 252)
        self.set_text_color(45, 55, 72)
        
        col_width_k = 60
        col_width_v = 130
        
        for k, v in specs.items():
            self.set_font("Helvetica", "B", 9)
            self.cell(col_width_k, 6, f"  {k}", border=1, fill=True)
            self.set_font("Helvetica", "", 9)
            self.cell(col_width_v, 6, f"  {v}", border=1, ln=True)
        self.ln(4)

def generate_report():
    pdf = VerifixiaReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # ── SECTION 1: INTRODUCTION & OVERVIEW ──
    pdf.chapter_title("1", "Detection Pipeline Overview")
    pdf.body_text(
        "Verifixia operates a multi-tier detection pipeline designed to maximize processing efficiency and accuracy. "
        "Each incoming request passes through automated pre-checks before hitting the core deep learning models. "
        "In the event of an infrastructure or PyTorch execution failure, the system seamlessly cascades down through "
        "alternative classifiers to ensure continuous uptime."
    )
    
    # Textual diagram representation
    pdf.set_font("Courier", "", 9)
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(26, 54, 93)
    diagram_text = (
        " [Upload/Stream] --> [Cartoon/Art Pre-check (PIL)] --(If Synthetic)--> [Fake (90%)]\n"
        "                           |\n"
        "                      (If Photographic)\n"
        "                           v\n"
        "                  [Tier 1: PyTorch Models] --------------> [Multi-Class / Binary prediction]\n"
        "                           |\n"
        "                     (If Error / Missing)\n"
        "                           v\n"
        "                  [Tier 2: scikit-learn SVM (HOG+Color)] -> [Prediction & Threat Score]\n"
        "                           |\n"
        "                     (If Error / Missing)\n"
        "                           v\n"
        "                  [Tier 3: Contrast & Brightness Heuristic] -> [Final Fallback Prediction]"
    )
    pdf.multi_cell(0, 4, diagram_text, border=1, fill=True)
    pdf.ln(5)
    
    # ── SECTION 2: MULTI-CLASS IMAGE DETECTOR ──
    pdf.chapter_title("2", "Multi-Class Image Detector (Primary PyTorch Model)")
    pdf.body_text(
        "The primary deep learning model is a custom 3-class classifier designed to separate authentic media from "
        "different types of manipulations (AI synthesis vs. classical splicing/swaps)."
    )
    
    multiclass_specs = {
        "Architecture": "Custom ResNet backbone integrated with Squeeze-and-Excitation (SE) Attention",
        "Target Classes": "3 classes: 0 = Real, 1 = Deepfake, 2 = AI-Generated",
        "Total Parameters": "12,975,675",
        "Trainable Parameters": "12,975,675",
        "Input Dimensions": "3 channels x 299 x 299 pixels",
        "Classification Head": "Linear bottleneck head (1024 -> 512 -> 256 -> 3 classes) with Dropout",
        "Peak Validation Accuracy": "89.09% (trained over 80 epochs)",
        "Training Weight File": "models/multiclass_detector.pth"
    }
    pdf.spec_table(multiclass_specs)
    
    # ── SECTION 3: BINARY IMAGE DEEPFAKE DETECTOR ──
    pdf.chapter_title("3", "Binary Image Deepfake Detector (Xception Fallback)")
    pdf.body_text(
        "For binary verification, the platform contains an alternative Xception-based deep learning model "
        "trained strictly on binary classification. It utilizes a similar custom attention architecture."
    )
    
    binary_specs = {
        "Architecture": "Xception-based CNN / DeepfakeDetector with SE Attention blocks",
        "Target Output": "Binary prediction: Real (0) or Fake/Deepfake (1)",
        "Total Parameters": "12,975,161",
        "Input Dimensions": "3 channels x 299 x 299 pixels",
        "Classification Head": "Sequential Dense layers (1024 -> 512 -> 256 -> 1) with Sigmoid activation",
        "Training Weight File": "models/xception_deepfake.pth"
    }
    pdf.spec_table(binary_specs)
    
    # Page Break for layout cleanliness
    pdf.add_page()
    
    # ── SECTION 4: VIDEO DEEPFAKE DETECTOR ──
    pdf.chapter_title("4", "Video Deepfake Detector (DeeperForensics Pipeline)")
    pdf.body_text(
        "The video detection pipeline handles dynamic content using a frame-sample aggregation strategy, "
        "integrating facial localization and temporal analysis."
    )
    
    video_specs = {
        "Face Extractor": "OpenCV Haar Cascades (frontal face detection)",
        "Frame Sampling": "Extracts up to 25 evenly-spaced frames from the video sequence",
        "Decision Threshold": "0.5 (loaded from models/deeperforensics_info.json)",
        "Evaluation Dataset": "DeeperForensics-1.0 local evaluation subset",
        "Training Epochs": "30 epochs",
        "Best Validation Accuracy": "100.0% (on local validation split)",
        "Aggregation Function": "Frame-wise average of binary confidence scores"
    }
    pdf.spec_table(video_specs)
    
    # ── SECTION 5: FALLBACK SYSTEMS ──
    pdf.chapter_title("5", "Fallback Classifiers (Tiers 2 & 3)")
    pdf.body_text(
        "To guarantee high system availability and zero downtime on low-resource environments (such as basic CPU nodes), "
        "Verifixia integrates non-deep-learning classifiers."
    )
    
    pdf.section_title("Tier 2 Fallback: scikit-learn SVM")
    pdf.body_text(
        "A classical Support Vector Machine (SVM) model trained on hand-crafted visual features. It runs without a GPU."
    )
    
    svm_specs = {
        "Classifier Type": "Support Vector Machine (SVM) with RBF (Radial Basis Function) Kernel",
        "Feature Set": "Histogram of Oriented Gradients (HOG) + RGB/YCbCr color histograms + FFT spectrum",
        "Dimensionality": "Concatenated multi-space feature vector normalized via StandardScaler",
        "Input Size": "128 x 128 pixels (downscaled for fast feature extraction)",
        "Saved Package File": "models/deepfake_sklearn.pkl"
    }
    pdf.spec_table(svm_specs)
    
    pdf.section_title("Tier 3 Fallback: Heuristic Analysis")
    pdf.body_text(
        "If both PyTorch and scikit-learn modules fail, the system runs raw pixel statistics using standard Pillow libraries. "
        "It evaluates absolute brightness offsets and standard deviation contrast measurements, blending them with a "
        "slight random fluctuation to construct a provisional confidence indicator."
    )
    
    pdf.section_title("Cartoon & Synthetic Art Detector (Pre-processing Filter)")
    pdf.body_text(
        "Before feeding an image to PyTorch/SVM, a dedicated heuristic filter checks color variety and channel uniformity. "
        "Since realistic deepfake models perform poorly on illustrated media, drawings are immediately classified as "
        "Synthetic/Fake with 90% confidence, saving CPU/GPU resources."
    )
    
    # Output the PDF
    output_path = os.path.normpath("models/verifixia_model_specs.pdf")
    pdf.output(output_path)
    print(f"PDF successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_report()
