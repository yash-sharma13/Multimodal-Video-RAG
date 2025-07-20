# 🎥 Video RAG: Extract, Understand, and Chat with Your Videos!

Welcome to **Multi Modal Video RAG**, a powerful Retrieval Augmented Generation (RAG) system designed to interactively query and extract information from YouTube videos. This project combines advanced video processing with multimodal AI capabilities to provide a seamless chat experience over video content. 

## ✨ Features

*   **YouTube Video Ingestion**: Easily download and process YouTube videos.
*   **Multimodal Data Extraction**: Automatically extract key frames, audio, and transcribe spoken content from videos.
*   **Intelligent Indexing**: Utilizes LanceDB for efficient storage and retrieval of both textual and image embeddings.
*   **Google Generative AI Integration**: Leverages `gemini-pro-vision` for powerful multimodal understanding and response generation.
*   **Interactive Chat Interface**: A user-friendly web interface (built with Next.js) to chat with your video content.
*   **Persistent Storage**: Maintains processed video data and indexes for quick access. 

## 🛠️ Technologies Used

**Backend (Python)**:
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

**Frontend (Next.js)**:
*   Next.js
*   React
*   Tailwind CSS
*   TypeScript 

## 🚀 Getting Started

Follow these steps to set up and run the Video RAG project on your local machine.

### Prerequisites

Ensure you have the following installed:

*   **Python 3.9+**
*   **Node.js (LTS recommended)** and **npm** or **pnpm**
*   **Git**

### Installation

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/YOUR_USERNAME/Video-RAG.git
    cd Video-RAG
    ```
    (Replace `YOUR_USERNAME` with your GitHub username or the organization's name.)

2.  **Backend Setup (Python)**:

    Navigate to the root directory of the cloned repository if you are not already there.

    ```bash
    cd Video-RAG
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

    Create a `.env` file in the root directory (`Video-RAG/.env`) and add your Google API key. You can obtain a Google API key from the [Google AI Studio](https://aistudio.google.com/): 

    ```
    GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'
    ```
    *Make sure to replace `YOUR_GOOGLE_API_KEY` with your actual API key.*

3.  **Frontend Setup (Next.js Chatbot)**:

    Navigate to the `ai-chatbot` directory:

    ```bash
    cd ai-chatbot
    ```

    Install the Node.js dependencies (using npm or pnpm):

    ```bash
    npm install # or pnpm install
    ```

    Create a `.env.local` file in the `ai-chatbot` directory (`Video-RAG/ai-chatbot/.env.local`) and add your NextAuth.js secret and Google Client ID/Secret. Follow the instructions in `ai-chatbot/.env.example` for the required variables. 

    Example `.env.local`:
    ```
    AUTH_SECRET=YOUR_NEXTAUTH_SECRET_HERE
    AUTH_GOOGLE_ID=YOUR_GOOGLE_CLIENT_ID
    AUTH_GOOGLE_SECRET=YOUR_GOOGLE_CLIENT_SECRET
    ```
    *Generate a strong secret for `AUTH_SECRET`.*

### Running the Application

1.  **Start the Backend Server (Python)**:

    From the root directory of the repository (`Video-RAG`), with your Python virtual environment activated:

    ```bash
    python src/server.py
    ```
    This will start the FastAPI server. You will be prompted in the terminal to enter a YouTube URL to begin processing.

2.  **Start the Frontend Chatbot (Next.js)**:

    In a new terminal, navigate to the `ai-chatbot` directory:

    ```bash
    cd ai-chatbot
    ```

    Start the Next.js development server:

    ```bash
    npm run dev # or pnpm run dev
    ```

    The frontend application will be accessible at `http://localhost:3000` (or another port if 3000 is in use). 

## 💡 Usage

Once both the backend and frontend servers are running:

1.  **Process a Video**: In the terminal where the Python backend is running, you will be prompted to `Enter the YouTube video URL:`. Paste a valid YouTube link and press Enter. The system will download the video, extract relevant data (frames, audio, transcript), and build its multimodal index.
2.  **Interact via Chatbot**: Open your web browser and navigate to `http://localhost:3000`. You can then log in (if authentication is configured in `ai-chatbot`) and start asking questions about the processed video content. The chatbot will retrieve information from the video and provide responses.

## 📁 Project Structure

Here's a high-level overview of the main directories:

*   `src/`: Contains the core Python backend logic for video processing, RAG, and the FastAPI server.
*   `ai-chatbot/`: Houses the Next.js frontend application, providing the interactive chat interface.
*   `output/`: Stores processed video data, extracted images, transcripts, and LanceDB vector stores.
*   `model_cache/`: Stores cached models used by `transformers` and `faster-whisper`.
*   `.venv/`: (Python) Virtual environment for backend dependencies. 

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📌 Tips for GitHub Push

When pushing this project to GitHub, it's crucial to exclude certain files and directories to maintain security, keep your repository clean, and avoid pushing unnecessary large files. Here's what you should typically ignore:

*   **`.env` and `.env.local` files**: These files contain sensitive API keys and secrets. **Never push them to public repositories.** Your `.gitignore` should include these entries.
*   **Virtual Environments (`.venv/`, `node_modules/`)**: These directories contain installed dependencies specific to your local environment. They can be very large and are not needed in the repository, as users can install them using `requirements.txt` and `package.json`.
    *   For the Python backend: Add `.venv/` to your `.gitignore`.
    *   For the Next.js frontend: The `ai-chatbot/node_modules/` directory should already be ignored by `ai-chatbot/.gitignore`.
*   **`output/`**: This directory contains processed video data, extracted images, and LanceDB data. These are generated files and can be large. It's best to exclude them.
*   **`model_cache/`**: This directory stores cached AI models. These can be very large and are typically downloaded on demand. Exclude this from your repository.
*   **`backup/`**: As you mentioned, this folder contains previous versions of code. It should definitely be excluded from the main repository.
*   **`.next/`**: For the Next.js application, this directory is a build output and should be ignored.

**Ensure your `.gitignore` file (in the root of `Video-RAG`) includes entries similar to these:**

```
# Environment variables
.env

# Python virtual environment
.venv/

# Generated outputs
output/
model_cache/

# Backup folder
backup/

# Next.js build output
/ai-chatbot/.next/
/ai-chatbot/node_modules/
/ai-chatbot/.env.local

# If you have specific IDE files (e.g., .idea/, .vscode/) you might want to add them
```

By following these tips, you can ensure a cleaner, more secure, and efficient GitHub repository for your Video RAG project! 