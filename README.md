# 🎥 Multimodal Video RAG: Extract, Understand, and Chat with Your Videos!

Welcome to **Multi Modal Video RAG**, a powerful Retrieval Augmented Generation (RAG) system designed to interactively query and extract information from YouTube videos. This project combines advanced video processing with multimodal AI capabilities to provide a seamless chat experience over video content. 

## ✨ Features

*   **YouTube Video Ingestion**: Easily download and process YouTube videos.
*   **Multimodal Data Extraction**: Automatically extract key frames, audio, and transcribe spoken content from videos.
*   **Intelligent Indexing**: Utilizes LanceDB for efficient storage and retrieval of both textual and image embeddings.
*   **Google Generative AI Integration**: Leverages `gemini-pro-vision` for powerful multimodal understanding and response generation.
*   **Persistent Storage**: Maintains processed video data and indexes for quick access. 

## 🛠️ Technologies Used

*   Python
*   `llama-index`: For RAG framework.
*   `LanceDB`: Vector store for efficient data indexing and retrieval.
*   `pytubefix`: For YouTube video downloading.
*   `moviepy`: For video and audio processing.
*   `faster-whisper`: For efficient audio transcription.
*   `pytesseract` and `Pillow`: For image OCR (Optical Character Recognition).
*   `sentence-transformers`: For generating embeddings.
*   `transformers`: Hugging Face Transformers library.
*   `google-generativeai`: For interacting with Google's Gemini models.
*   `FastAPI` and `Uvicorn`: For building the API server.

## 🚀 Getting Started

Follow these steps to set up and run the Video RAG project on your local machine.

### Prerequisites

Ensure you have the following installed:

*   **Python 3.9+**
*   **Git**

### Installation

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/YOUR_USERNAME/Multimodal-Video-RAG.git
    cd Multimodal-Video-RAG
    ```
    (Replace `YOUR_USERNAME` with your GitHub username or the organization's name.)

2.  **Setup (Python)**:

    Navigate to the root directory of the cloned repository if you are not already there.

    ```bash
    cd Multimodal-Video-RAG
    ```

    Create a Python virtual environment and activate it:

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

    Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Create a `.env` file in the root directory (`Multimodal-Video-RAG/.env`) and add your Google API key. You can obtain a Google API key from the [Google AI Studio](https://aistudio.google.com/): 

    ```
    GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'
    ```
    *Make sure to replace `YOUR_GOOGLE_API_KEY` with your actual API key.*


### Running the Application

1.  **Start the application (Python)**:

    From the root directory of the repository (`Multimodal-Video-RAG`), with your Python virtual environment activated:

    ```bash
    python src/multimodal_rag.py
    ```
    This will start the application locally in the terminal. You will be prompted in the terminal to enter a YouTube URL to begin processing.

## 💡 Usage

Once the application is running:

1.  **Process a Video**: In the terminal where the Python backend is running, you will be prompted to `Enter the YouTube video URL:`. Paste a valid YouTube link and press Enter. The system will download the video, extract relevant data (frames, audio, transcript), and build its multimodal index.

## 📁 Project Structure

Here's a high-level overview of the main directories:

*   `src/`: Contains the core Python backend logic for video processing, RAG, and the FastAPI server.
*   `output/`: Stores processed video data, extracted images, transcripts, and LanceDB vector stores.
*   `model_cache/`: Stores cached models used by `transformers` and `faster-whisper`.
*   `.venv/`: (Python) Virtual environment for backend dependencies. 

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📌 Tips for GitHub Push

When pushing this project to GitHub, it's crucial to exclude certain files and directories to maintain security, keep your repository clean, and avoid pushing unnecessary large files. Here's what you should typically ignore:

*   **`.env` files**: These files contain sensitive API keys and secrets. **Never push them to public repositories.** Your `.gitignore` should include these entries.
*   **Virtual Environments (`.venv/`, `node_modules/`)**: These directories contain installed dependencies specific to your local environment. They can be very large and are not needed in the repository, as users can install them using `requirements.txt` and `package.json`.
    *   For the Python application: Add `.venv/` to your `.gitignore`.
*   **`output/`**: This directory contains processed video data, extracted images, and LanceDB data. These are generated files and can be large. It's best to exclude them, or keep the empty folder structure by adding `.gitkeep` file in each folder.
*   **`model_cache/`**: This directory stores cached AI models. These can be very large and are typically downloaded on demand. Again, you can keep the empty folder structure by adding `.gitkeep` file in each folder.

**Ensure your `.gitignore` file (in the root of `Multimodal- Video-RAG`) includes entries similar to these:**

```
# Environment variables
.env

# Python virtual environment
.venv/

```

NOW YOU ARE ALL SET! YOU CAN USE THE RAG APPLICATION OR TRY OUT DIFFERENT APPROACHES OR WAYS TO CONTRIBUTE AS WELL. CHEERS ✨
