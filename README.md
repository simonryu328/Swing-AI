# Swing-AI

<p align="center">
  <img src="images/demo.gif" alt="Streamlit demo" />
</p>

# Golf Instruction Multimodal RAG Chatbot

## Project Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) chatbot focused on golf instruction, emulating the teaching style of legendary golfer Ben Hogan. The chatbot leverages state-of-the-art natural language processing and computer vision techniques to provide accurate, context-aware responses along with relevant visual aids.

## Key Features

1. **Advanced Text Processing**
   - Utilized the `StatisticalChunker` from the `semantic_chunkers` library for intelligent text segmentation.
   - Implemented sophisticated text splitting methods to create meaningful chunks for embedding.

2. **AI-Powered Image Labeling**
   - Employed Claude, an advanced AI model, to generate detailed descriptions of golf-related images.
   - These AI-generated descriptions enhance the quality and relevance of image retrieval.

3. **Multimodal Embeddings**
   - Created embeddings for both text and AI-labeled images using CLIP (Contrastive Language-Image Pre-Training).
   - Employed FAISS for efficient similarity search on image embeddings.

4. **Vector Storage and Retrieval**
   - Used Chroma as a vector store for text embeddings, enabling fast and relevant information retrieval.
   - Implemented a custom retrieval mechanism for matching relevant images to text responses.

5. **RAG Pipeline**
   - Developed a Retrieval-Augmented Generation pipeline using LangChain.
   - Integrated OpenAI's GPT model for high-quality text generation based on retrieved context.

6. **Prompt Engineering**
   - Crafted detailed prompts to guide the AI in responding in Ben Hogan's distinctive style.
   - Designed prompts to seamlessly incorporate diagram descriptions into responses.

7. **Streamlit Web Application**
   - Created an interactive web interface using Streamlit for easy user interaction.
   - Implemented real-time response generation and relevant image display.

## Technical Details

- **Text Processing**: Used `StatisticalChunker` for creating semantically meaningful text chunks.
- **Image Labeling**: Leveraged Claude AI for generating descriptive labels for golf-related images.
- **Embeddings**: Utilized CLIP for creating joint embeddings of text and AI-labeled images.
- **Similarity Search**: Implemented FAISS for fast and efficient similarity search in high-dimensional spaces.
- **Vector Storage**: Employed Chroma for storing and retrieving text embeddings.
- **Language Model**: Utilized OpenAI's GPT for generating human-like responses.
- **Web Framework**: Built with Streamlit for a responsive and user-friendly interface.

## Innovative Aspects

- **AI-Assisted Data Preparation**: The use of Claude for image labeling represents an innovative approach to preparing multimodal data for embedding and retrieval.
- **Semantic Chunking**: The application of statistical chunking methods ensures that text is split into meaningful, context-preserving segments.
- **Multimodal Retrieval**: By combining text and image embeddings, the system can provide more comprehensive and relevant responses to user queries.

## Future Enhancements

- Implement conversation history management for context-aware multi-turn interactions.
- Expand the knowledge base to cover a wider range of golf instruction topics.
- Optimize retrieval and generation pipeline for improved response times.
- Implement user feedback mechanism for continuous improvement of the chatbot.
- Explore fine-tuning of language models on golf-specific data for even more accurate responses.

## Getting Started

- See `notebooks/rag.ipynb` for the retrieval pipeline.
- To run the streamlit app, create a virtual environment with python version <= 3.11, install the `app/requirements.txt` and run `streamlit run app.py`.

## Copyright Notice and Legal Disclaimer

This project uses content derived from published golf instruction materials. The use of this content is intended for educational and research purposes only.

Important points to note:

1. This project is not affiliated with, endorsed by, or connected to Ben Hogan, his estate, or any official Ben Hogan branded entities.
2. The content used in this project may be subject to copyright. Users should be aware that the redistribution or commercial use of this content may infringe on copyright laws.
3. This project is a proof-of-concept and should not be used for commercial purposes without proper licensing and permissions from copyright holders.
4. If you are a copyright holder and believe that your work has been used inappropriately in this project, please contact [simonryu328@gmail.com] to address the issue.

Users are advised to seek proper legal advice before using this project or its outputs for any purpose other than personal, educational use.

## Ethical Considerations

While this project aims to innovate in the field of AI and golf instruction, it's important to consider the ethical implications of using copyrighted material. Future iterations of this project should explore:

1. Obtaining proper permissions or licenses for copyrighted content.
2. Developing original content in collaboration with golf professionals.
3. Using only publicly available, open-source golf instruction materials.
