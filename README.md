# 📜 Haikify

A simple web app that generates elegant, original haikus inspired by three images you provide.  
Built with **Streamlit** for the frontend, 
**TensorFlow** for image classification, 
and AI-powered haiku generation.
---

## ✨ Features
- **AI-generated haikus** following the traditional 5–7–5 structure.
- Inspired by three user-provided pictures.
- Uses two different models to provide the best possible haiku.

---

## 🛠 Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**:
  - Python
  - TensorFlow for image classification
  - Multiple LLMs for haiku generation and evaluation
- **Async architecture** for fast and scalable responses.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/michalwegiel/Haikify.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
streamlit==1.50.0
tensorflow==2.20.0
opencv-python==4.12.0.88
openai-agents==0.4.2

pillow~=11.3.0
numpy~=2.2.6
keras~=3.11.3
python-dotenv~=1.2.1
```

### 3. Run the app
```bash
streamlit run main.py
```

---

## 📂 Project Structure
```
Haikify/
├── assets
├── main.py             
├── haiku_agent.py
├── image_classify.py
├── requirements.txt
└── README.md
```

---

## 🖥 How It Works
1. **User Uploads Images**: Provide three pictures in **JPG** or **PNG** format through the Streamlit UI.
2. **Image Classification**: **TensorFlow** processes and classifies each image to identify key concepts or themes.
3. **Haiku Generation**:
   - Two separate **LLMs** generate haikus inspired by the classified concepts.
4. **Quality Selection**: An additional **LLM** evaluates both haikus and decides which one is better based on:
   - Creativity
   - Adherence to the 5–7–5 structure
   - Thematic relevance
5. **Display**: The selected haiku is shown in a clean, minimal interface.

---

## ✅ Example Usage
- Upload: `forest.jpg`, `river.png`, `mountain.jpg`
- Output:
```
Silent mist drifts low
Whispers curl around cold stone—
Echo fades to dusk
```
