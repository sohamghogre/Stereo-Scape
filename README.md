# 🎉 Welcome to StereoScape 🎉

StereoScape is an innovative project that converts 2D images and videos into 3D models using the NeRF (Neural Radiance Fields) model. This web application has a user-friendly frontend built with SvelteKit and TypeScript, and a powerful backend using Python Flask, COLMAP, and TensorFlow.

## 🚀 Features
- Convert 2D images/videos into 3D models
- User-friendly interface with fluid animations
- High-quality 3D reconstruction using Tiny NeRF model
- Applications in gaming, medical imaging, research, VR/AR, and more

## 🛠️ Technologies Used

### Frontend
- SvelteKit
- TypeScript
- Tailwind CSS
- WebGL animations

### Backend
- Python Flask
- COLMAP
- TensorFlow
- MongoDB

## ⚙️ Installation and Setup

### Backend Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/LittleZabi/stereo-scape.git
   cd stereo-scape/backend
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. **Install the required Python libraries:**
   ```sh
    pip install -r requirements.txt`
4. **Start the backend server:**
   ```sh
    python app.py`\
### Frontend Setup
1. **Navigate to the frontend directory:**
   ```sh
   cd ../web
2. **Install the required npm packages:**
   ```sh
   npm install
3. **Start the frontend server:**
   ```sh
   npm run dev
### 📂 Project Structure
  - `backend/`: Contains the Flask application and related Python scripts.
  - `web/`: Contains the SvelteKit frontend application.
  - `requirements.txt`: Lists the Python libraries required for the backend.
  - `package.json`: Lists the npm packages required for the frontend.

### 📋 Usage
  - Upload Images/Videos: Use the user-friendly interface to upload 2D images or videos.
  - Processing: The backend processes the images/videos, extracts features using COLMAP, and creates a dataset.
  - 3D Reconstruction: The NeRF model processes the dataset to generate high-quality 3D models, viewable from multiple angles.


### Project Demo and implementation

Watch the demo of the StereoScape project below:

[![Watch the video](https://img.youtube.com/vi/9jEcSEJdKZ0/maxresdefault.jpg)](https://youtu.be/9jEcSEJdKZ0)
Click the image above to watch the demo video.

### 🌟 Contributions
Feel free to contribute to this project by submitting issues or pull requests. Your contributions are highly appreciated!

📄 License
This project is licensed under the MIT License.

---

Made with ❤️ by [LittleZabi](https://github.com/LittleZabi)
