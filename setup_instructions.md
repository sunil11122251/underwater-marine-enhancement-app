# Setup Instructions

Follow the steps below to run the **Underwater Marine Enhancement App** on your system.

---

## 1. Clone the Repository

Open terminal or command prompt and run:

git clone https://github.com/sunil11122251/underwater-marine-enhancement-app.git

cd underwater-marine-enhancement-app

---

## 2. Create a Virtual Environment (Optional but Recommended)

Create a virtual environment:

python -m venv venv

Activate the environment:

### Windows

venv\Scripts\activate

### Mac/Linux

source venv/bin/activate

---

## 3. Install Required Dependencies

Install all required Python libraries using:

pip install -r requirements.txt

---

## 4. Ensure Model Weights are Available

Make sure the trained model file is present in the `weights` folder.

Example:

weights/adgol_best.pth

---

## 5. Run the Application

Start the application using:

python app.py

---

## 6. Upload an Underwater Image

Open the application interface and upload an underwater image to enhance.

---

## 7. View Enhanced Output

The system will process the input image and display the enhanced underwater image.

---

## Requirements

- Python 3.8 or higher
- PyTorch
- OpenCV
- NumPy

All required dependencies are listed in `requirements.txt`.
