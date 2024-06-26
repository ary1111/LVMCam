import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    load_in_4bit=True
) 

def main():
    st.title("Webcam Stream with Llava 4bit Chat")

    # Button to take a picture
    img_file_buffer = st.camera_input("Take a picture")

    # Create a text input for the user prompt
    st.header("Chat with LlavaNext")
    user_input = st.text_input("Enter your prompt here")

    # Button to prompt the model using the image and user input
    if st.button("Send Prompt"):
        if img_file_buffer:
            img = Image.open(img_file_buffer)            
            prompt = f"[INST] <image>\n{user_input}? [/INST]"
            inputs = processor(prompt, img, return_tensors="pt").to("cuda:0")

            output = model.generate(**inputs, max_new_tokens=100)
            response = processor.decode(output[0], skip_special_tokens=True)
            print(response)
            st.write("Response: ", response)
        else:
            st.error("Please take a picture first")

if __name__ == "__main__":
    main()
