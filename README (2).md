📦 Stereo-Scape
Stereo-Scape is a 3D scene reconstruction project that uses stereo vision techniques to generate depth maps and point clouds from pairs of stereo images. It aims to provide a simple, modular, and extensible framework for experimenting with stereo image processing, disparity estimation, and 3D model generation.

✨ Features
📷 Load stereo image pairs (left and right views)

🛠️ Perform stereo rectification

📏 Compute disparity maps

🌍 Generate 3D point clouds

🎨 Visualize disparity and depth outputs

🔧 Easy-to-modify pipeline for custom stereo algorithms
> To deploy your app, you may need to install an [adapter](https://kit.svelte.dev/docs/adapters) for your target environment.

stereo-scape/
│
├── data/              # Sample stereo image pairs
├── output/            # Generated outputs (disparity maps, point clouds)
├── src/               # Source code files
│    ├── disparity.py  # Disparity map generation
│    ├── stereo.py     # Stereo image rectification and processing
│    ├── utils.py      # Helper functions
│    └── config.py     # Configuration settings
├── requirements.txt   # Python dependencies
└── README.md          # Project description and setup instructions

![Screenshot 2025-04-23 184909](https://github.com/user-attachments/assets/efb8c16e-b9fa-4910-ae77-017f418b02a5)
![Screenshot 2025-04-22 162847](https://github.com/user-attachments/assets/c8f5c876-783c-46d1-b8d2-dce500f659a5)
![Screenshot 2025-04-22 163043](https://github.com/user-attachments/assets/590ad511-94a8-45d0-91e2-4f5eef76d225)
![Screenshot 2025-04-25 222847](https://github.com/user-attachments/assets/abd60d66-732f-4b63-8490-97a15124ed34)
![Screenshot 2025-04-26 105252](https://github.com/user-attachments/assets/776a6265-3df0-471a-9b10-44dc8b8433ac)








🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/sohamghogre/Stereo-Scape
cd stereo-scape
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Make sure you have OpenCV (opencv-python) and NumPy installed.

3. Run the Code
bash
Copy
Edit
python src/stereo.py
The program will:

Load sample stereo images

Compute the disparity map

Reconstruct a 3D point cloud

Save the results into the output/ folder

🛠️ Built With
Python

OpenCV

NumPy

Matplotlib (for visualizations)
