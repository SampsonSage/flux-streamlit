# app.py
import torch
import streamlit as st
from diffusers import FluxPipeline
import base64
from io import BytesIO
from PIL import Image
import time

# Set page config for dark theme
st.set_page_config(
    page_title="FLUX Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dracula theme
st.markdown("""
<style>
    /* Dracula theme colors */
    :root {
        --draculaBg: #282a36;
        --draculaFg: #f8f8f2;
        --draculaPurple: #bd93f9;
        --draculaPink: #ff79c6;
        --draculaGreen: #50fa7b;
    }
    
    .stApp {
        background-color: var(--draculaBg);
        color: var(--draculaFg);
    }
    
    .stTextInput > div > div {
        background-color: #44475a;
        color: var(--draculaFg);
        border-radius: 10px;
    }
    
    .stButton > button {
        background-color: var(--draculaPurple);
        color: var(--draculaBg);
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--draculaPink);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(189, 147, 249, 0.3);
    }
    
    .stSlider > div > div {
        background-color: var(--draculaPurple);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    .generating {
        animation: pulse 2s infinite;
        color: var(--draculaGreen);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe.enable_model_cpu_offload()
    return pipe

def generate_image(pipe, prompt, guidance_scale, height, width, steps):
    with torch.cuda.amp.autocast():
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=steps,
        ).images[0]
    return image

def main():
    st.title("🎨 FLUX Image Generator")
    
    # Initialize session state
    if 'image_history' not in st.session_state:
        st.session_state.image_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("Generation Settings")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 3.5, 0.5)
        height = st.select_slider("Height", options=[512, 768, 1024], value=768)
        width = st.select_slider("Width", options=[512, 768, 1024, 1360], value=1360)
        steps = st.slider("Inference Steps", 20, 100, 50, 5)
    
    # Main content
    prompt = st.text_area("Enter your prompt:", "a tiny astronaut hatching from an egg on the moon",
                         help="Describe the image you want to generate")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate", use_container_width=True):
            try:
                with st.spinner("Loading model..."):
                    pipe = load_model()
                
                progress_text = st.empty()
                progress_text.markdown("""
                    <div class="generating">
                        🎨 Generating your masterpiece...
                    </div>
                """, unsafe_allow_html=True)
                
                start_time = time.time()
                image = generate_image(pipe, prompt, guidance_scale, height, width, steps)
                end_time = time.time()
                
                # Convert to bytes for display
                buf = BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                # Add to history
                st.session_state.image_history.insert(0, {
                    'prompt': prompt,
                    'image': byte_im,
                    'settings': f"Guidance: {guidance_scale}, Steps: {steps}"
                })
                
                progress_text.empty()
                st.success(f"Generated in {end_time - start_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
    
    # Display history
    st.header("Generated Images")
    for i, item in enumerate(st.session_state.image_history):
        with st.expander(f"Image {i+1}: {item['prompt'][:50]}...", expanded=(i==0)):
            st.image(item['image'], caption=item['settings'])
            
            # Download button
            col1, col2 = st.columns([3, 1])
            with col2:
                b64 = base64.b64encode(item['image']).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="generated_image_{i}.png">Download Image</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
