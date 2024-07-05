import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import dotenv
from PIL import Image

dotenv.load_dotenv()
st.title("Swing AI")


import numpy as np
import torch
import faiss
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# Change the working directory to the directory where app.py is located
os.chdir(os.path.dirname(__file__))

# Now the relative path will work
df = pd.read_csv("enhanced_image_descriptions.csv")
image_dir = "../data/images/"


from torch.utils.data import Dataset

class GolfDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name="openai/clip-vit-base-patch16"):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        description = self.dataframe.iloc[idx, 1]
        description_tensor = self.tokenizer(description, truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
        return description_tensor

# Create the dataset
golf_dataset = GolfDataset(dataframe=df)

def create_embeddings(dataset, model, device):
    embeddings = []
    for i in range(len(dataset)):
        description_tensor = dataset[i].unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            embedding = model.get_text_features(description_tensor)[0].detach().cpu().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

# Create embeddings
embeddings = create_embeddings(golf_dataset, model, device)

# Example of how to access data
print(embeddings.shape)  # Should be (number_of_samples, embedding_dim)

# Step 2: Initialize FAISS index
dimension = embeddings.shape[1]  # Get the dimension of embeddings
index = faiss.IndexFlatL2(dimension)  # Create FAISS index
index.add(embeddings)  # Add embeddings to the index

# Step 3: Query the dataset with a text prompt
def query_with_prompt(prompt, model, tokenizer, index, dataframe, image_dir, k=1):
    # Get prompt embedding
    prompt_embedding = (
        model.get_text_features(**tokenizer([prompt], return_tensors="pt", truncation=True).to(device))[0]
        .detach()
        .cpu()
        .numpy()
    )

    # Find nearest embeddings
    distances, indices = index.search(np.array([prompt_embedding]), k)
    matched_indices = indices[0]

    print(distances)

    # Retrieve matched image paths and descriptions
    matched_images = [os.path.join(image_dir, dataframe.iloc[idx, 0]) for idx in matched_indices]
    matched_descriptions = [dataframe.iloc[idx, 1] for idx in matched_indices]

    return matched_images, matched_descriptions

# Step 4: Display the matched image
def display_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width = 200
    ratio = width / float(image.size[0])
    height = int((float(image.size[1]) * float(ratio)))
    img = image.resize((width, height), Image.Resampling.LANCZOS)
    img.show()



# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OllamaEmbeddings(model='nomic-embed-text'), collection_name="agentic-chunks")
retriever = vectorstore.as_retriever()

prompt_template = """
### Ben Hogan Chatbot Instruction

**Role:** You are Ben Hogan, the legendary golfer, known for your precise ball-striking, meticulous approach to the game, and deep understanding of golf fundamentals.

**Tone:** Calm, authoritative, encouraging, and reflective of the 1950s era.

**Objective:** Answer user questions about golf, specifically focusing on golf swing techniques, mindset, philosophy, and love for the game, based on the book "Five Lessons: The Fundamentals of Golf."

#### Specific Instructions

**1. Provide Expert Golf Swing Advice:**
   - Break down the golf swing into its fundamental components: grip, stance and posture, the first part of the swing, and the second part of the swing.
   - Use detailed explanations from the book "Five Lessons: The Fundamentals of Golf."
   - Example:
     - User: "How should I hold the club to improve my grip?"
     - The grip is the foundation of your swing. Hold the club in your fingers, not your palm. The V formed by your thumb and index finger should point towards your right shoulder. A proper grip ensures control and consistency.

**2. Share Mindset and Philosophy:**
   - Discuss the mental approach to golf, emphasizing focus, discipline, and the importance of practice.
   - Share personal anecdotes and insights from your career.
   - Example:
     - User: "What mindset should I have when approaching a difficult shot?"
     - "Golf is as much a mental game as it is a physical one. Approach each shot with confidence and focus. Visualize the perfect shot, trust your swing, and stay calm. Every challenge is an opportunity to improve."

**3. Encourage and Motivate:**
   - Provide motivational support and encouragement.
   - Reinforce the idea that improvement comes with practice and dedication.
   - Highlight the joy and fulfillment of playing golf.
   - Example:
     - User: "How important is practice in becoming a good golfer?"
     - "Practice is the bedrock of success in golf. Consistent, deliberate practice hones your skills and builds muscle memory. Dedication to practice will pay off on the course."

**4. Reflect on Philosophy and Love for Golf:**
   - Discuss your philosophy on golf and life.
   - Share why you love golf and what it means to you.
   - Example:
     - User: "What philosophy did you follow throughout your golf career?"
     - "My philosophy was simple: strive for perfection in every aspect of the game. Understand the fundamentals, work tirelessly to improve, and never settle for mediocrity. Golf is a journey of continuous learning and growth."

**Guidelines:**
- Always base responses on the teachings from "Five Lessons: The Fundamentals of Golf."
- Be informative, supportive, and engaging.
- Use quotes and references from your book to lend authenticity and depth to your answers.

Answer the question based on the following context:
{context}
Question: {question}
Now provide your answer. Do not wrap your response in quotes, just return the text itself.
"""

system_prompt = ChatPromptTemplate.from_template(prompt_template)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
chain = (
    {"context": retriever, "question": lambda x:x}
    | system_prompt
    | llm
    | StrOutputParser()
)



image_prompt_template = """
You are a golf instruction assistant. Combine the initial answer and diagram description into a concise, comprehensive response. Follow these rules strictly:

1. First say: Here is an illustration that can help you.
3. Briefly explain the illustration, adding only new information not covered in the initial answer.
4. Ensure that the transition between the initial answer and the diagram explanation feels natural and cohesive. You should explain how the diagram helps support your initial answer
4. Keep the total response under 100 words.
5. Do not use phrases like "As you can see", "But why is this important?", or "As I mentioned". Keep it concise
7. Maintain a direct, instructional tone throughout.

Example 1:
User Question: How important is follow-through in a golf swing?
Initial Answer: The follow-through is crucial in a golf swing. It ensures that you complete your swing with good balance and full extension, which helps maximize power and accuracy. A proper follow-through also indicates that you've maintained the correct swing path through impact.

Diagram Description: Diagram showing the correct spine angle at address, with the upper body tilted slightly forward and the lower body in an athletic stance.

Combined Response: Here is an illustration to support this concept. While this diagram shows the correct spine angle at address, it's highly relevant to achieving a proper follow-through. The slight forward tilt of the upper body and athletic stance of the lower body set up the foundation for a good follow-through. This initial posture allows you to maintain balance throughout the swing, enabling a full, extended follow-through. Remember, a good swing starts with proper setup, which directly impacts your ability to execute a effective follow-through.

Example 2:
User Question: What's the correct way to align my feet for a golf shot?
Initial Answer: For proper foot alignment in golf, your feet should be parallel to the target line, about shoulder-width apart. The line of your toes should point perpendicular to your target line. This alignment helps ensure that your body is properly positioned relative to your target, promoting a more accurate shot.

Diagram Description: Overhead view of a golfer's grip, showing the interlocking style where the pinky of the trailing hand interlocks with the index finger of the lead hand.

Combined Response: Here is an illustration that can help you. While this diagram focuses on the grip rather than foot alignment, it's important to understand how these elements work together. A proper grip, as shown in the diagram, complements correct foot alignment by ensuring that your hands are positioned to work in harmony with your body's alignment. The interlocking grip style depicted helps maintain a unified hand position throughout the swing, which is crucial for translating your correct foot alignment into an accurate shot. Remember, every aspect of your setup, from feet to grip, contributes to the overall success of your swing and shot accuracy.

Question: {question}
Initial Answer: {initial_answer}
Diagram Description: {diagram_description}

Now, provide your response for the given user question, initial answer, and diagram description:
"""
image_system_prompt = ChatPromptTemplate.from_template(image_prompt_template)
image_chain = (
    image_system_prompt
    | llm
    | StrOutputParser()
)
# result = chain.invoke({'question': question,'initial_answer': initial_result, 'diagram_description': matched_descriptions[0]})


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            st.image(message["image"], caption=message["image_caption"])
            st.markdown(message["image_content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response)

        # Image retrieval and display
        image_prompt = response
        matched_images, matched_descriptions = query_with_prompt(image_prompt, model, tokenizer, index, df, image_dir, k=1)
        if matched_images:
            image = Image.open(matched_images[0])
            st.image(image, caption=matched_descriptions[0])

            diagram_response = image_chain.invoke({'question': prompt,'initial_answer': response, 'diagram_description': matched_descriptions[0]})
            print(diagram_response)
            st.markdown(diagram_response)

    

    st.session_state.messages.append({"role": "assistant", "content": response, "image": image, "image_caption": matched_descriptions[0], "image_content": diagram_response})