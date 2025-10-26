import io

import streamlit as st
from PIL import Image


def main():
    st.set_page_config(page_title="Haikify", page_icon="assets/haikify_icon.jfif", layout="centered")

    st.title("Haikify")
    st.write("Upload three images to proceed.")

    cols = st.columns(3)
    uploaded_images = []

    for i, col in enumerate(cols, start=1):
        with col:
            st.write(f"**{i}{['st','nd','rd'][i-1]} image**")
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
        print(uploaded_images)
        print(len(uploaded_images))


if __name__ == "__main__":
    main()
