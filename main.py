import asyncio
import io

import streamlit as st
from PIL import Image

from haiku_agent import generate_best_haiku
from image_classify import classify_images


def main():
    st.set_page_config(page_title="Haikify", page_icon="assets/haikify_icon.jfif", layout="centered")

    col1, _ = st.columns([1, 4])
    with col1:
        st.image("assets/haikify_icon.jfif", width=60)

    st.title("Haikify")
    st.write("Upload three images to proceed.")

    cols = st.columns(3)
    uploaded_images = []

    for i, col in enumerate(cols, start=1):
        with col:
            st.write(f"**{i}{['st', 'nd', 'rd'][i - 1]} image**")
            file = st.file_uploader(
                "Choose an image…",
                type=["jpg", "jpeg", "png"],
                key=f"image_{i}",
                label_visibility="collapsed"
            )
            if file is not None:
                try:
                    img = Image.open(io.BytesIO(file.read())).convert("RGB")
                    st.image(img, caption="Preview", width="stretch")
                    uploaded_images.append(img)
                except Exception as e:
                    st.error(f"Invalid image file. Please try again. {e}")

    disabled = len(uploaded_images) != 3
    if disabled:
        st.info("Please upload all three images to enable Haikify.")

    btn = st.button("Haikify!", type="primary", disabled=disabled)
    if btn:
        with st.spinner("Writing Haiku..."):
            predictions = classify_images(uploaded_images)

            result = asyncio.run(generate_best_haiku(words=predictions))
            newline = "\n"
            st.markdown("---")
            st.markdown("### ✨ Your Haiku")
            st.markdown(f"""
                    <div style='font-size: 24px; line-height: 1.6; font-style: italic; color: #4B4B4B;'>
                        {result.replace(newline, '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")


if __name__ == "__main__":
    main()
