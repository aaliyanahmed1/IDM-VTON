"""
Simple Streamlit UI for IDM-VTON API
"""
import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="IDM-VTON Try-On", layout="wide")

st.title("🎨 IDM-VTON Virtual Try-On")
st.markdown("Upload a person image and garment image to generate virtual try-on result")

# API endpoint
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
API_ENDPOINT = f"{API_URL}/api/v1/tryon"

# Image upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("Person Image")
    person_file = st.file_uploader("Upload person image", type=["jpg", "jpeg", "png"], key="person")

with col2:
    st.subheader("Garment Image")
    garment_file = st.file_uploader("Upload garment image", type=["jpg", "jpeg", "png"], key="garment")

# Parameters
st.sidebar.subheader("Parameters")
denoise_steps = st.sidebar.slider("Denoise Steps", 10, 50, 20, help="Lower = faster, Higher = better quality")
auto_mask = st.sidebar.checkbox("Auto Mask", value=True, help="Automatically generate mask")
crop_image = st.sidebar.checkbox("Crop Image", value=False, help="Crop to optimal aspect ratio")
garment_description = st.sidebar.text_input("Garment Description (optional)", value="")

# Process button
if st.button("Generate Try-On", type="primary"):
    if person_file is None or garment_file is None:
        st.error("Please upload both person and garment images")
    else:
        try:
            with st.spinner("Processing... This may take 15-25 seconds"):
                # Prepare files
                files = {
                    'person_image': (person_file.name, person_file.getvalue(), person_file.type),
                    'garment_image': (garment_file.name, garment_file.getvalue(), garment_file.type)
                }
                
                data = {
                    'denoise_steps': denoise_steps,
                    'auto_mask': auto_mask,
                    'crop_image': crop_image
                }
                
                if garment_description:
                    data['garment_description'] = garment_description
                
                # Make request
                response = requests.post(API_ENDPOINT, files=files, data=data, timeout=180)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        # Display result
                        st.success("✅ Try-on completed successfully!")
                        
                        # Decode and display result image
                        result_image_data = base64.b64decode(result['result_image'])
                        result_image = Image.open(io.BytesIO(result_image_data))
                        
                        # Decode and display mask image
                        mask_image_data = base64.b64decode(result['mask_image'])
                        mask_image = Image.open(io.BytesIO(mask_image_data))
                        
                        # Show images side by side
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.image(person_file, caption="Person Image", use_container_width=True)
                        
                        with col2:
                            st.image(garment_file, caption="Garment Image", use_container_width=True)
                        
                        with col3:
                            st.image(result_image, caption="Result", use_container_width=True)
                        
                        # Download button
                        img_buffer = io.BytesIO()
                        result_image.save(img_buffer, format='PNG')
                        st.download_button(
                            label="Download Result",
                            data=img_buffer.getvalue(),
                            file_name="tryon_result.png",
                            mime="image/png"
                        )
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server may be processing. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the server is running.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Health check
if st.sidebar.button("Check API Health"):
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            st.sidebar.success("✅ API is healthy")
            st.sidebar.json(health)
        else:
            st.sidebar.error("API health check failed")
    except Exception as e:
        st.sidebar.error(f"Cannot connect to API: {str(e)}")

