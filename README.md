🌟 StereoScape - 3D Model Generator


Stereo-Scape is a 3D scene reconstruction project that uses stereo vision techniques to generate depth maps and point clouds from pairs of stereo images. It aims to provide a simple, modular, and extensible framework for experimenting with stereo image processing, disparity estimation, and 3D model generation.




![Screenshot 2025-04-23 184909](https://github.com/user-attachments/assets/24ca6808-29bb-4a58-821c-16d866edab15)
![Screenshot 2025-04-23 184925](https://github.com/user-attachments/assets/475f1a27-5a23-4b5a-a358-a303e798a37c)
![Screenshot 2025-04-23 184940](https://github.com/user-attachments/assets/1670e493-40b6-4151-8519-6373775903e2)
![Screenshot 2025-04-23 184954](https://github.com/user-attachments/assets/30e066af-fa7d-456f-a23e-b79ff59b02b7)
![Screenshot 2025-04-26 105252](https://github.com/user-attachments/assets/66b78984-e266-45e7-91af-68f0603a8f8d)


🎯 Overview
StereoScape leverages cutting-edge Neural Radiance Fields (NeRF) technology to:

Convert 2D images/videos into detailed 3D models
Generate high-resolution outputs (up to 4k)
Provide realistic rendering with accurate lighting and shadows
Enable novel view synthesis for immersive exploration

🚀 Key Features
Intuitive Upload Interface: Simple drag-and-drop functionality for images/videos
Real-time Processing: Watch your 3D model come to life with progress tracking
High-Quality Output: Generate detailed 3D models with realistic textures
Interactive Viewing: Explore your 3D models from any angle
Fluid Animations: Smooth user experience with beautiful visual effects
PSNR Tracking: Real-time Peak Signal-to-Noise Ratio monitoring
Multi-view Processing: Support for multiple image angles

💻 Technology Stack
Frontend
SvelteKit
TypeScript
Tailwind CSS
WebGL animations
Backend
Python Flask
COLMAP
TensorFlow
MongoDB

🛠️ Getting Started
Clone the repository:

git clone https://github.com/nilay2004/steroscape.git
Backend Setup:

cd stereo-scape/backend
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
python app.py
Frontend Setup:

cd ../web
npm install
npm run dev




