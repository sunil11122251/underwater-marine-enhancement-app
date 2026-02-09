import streamlit as st
import cv2
import numpy as np
import torch
from model import ADGOL
from utils import *
from metrics import compute_metrics

# =====================================================
st.set_page_config(
    page_title="Underwater Enhancement",
    page_icon="🌊",
    layout="centered"
)

st.title("🌊 Adaptive Image Enhancement Techniques for Underwater Marine Visuals")
# =====================================================

@st.cache_resource
def load_model():
    model = ADGOL()
    model.load_state_dict(
        torch.load("weights/adgol.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

uploaded = st.file_uploader(
    "Upload Underwater Image",
    type=["jpg", "png", "jpeg"]
)

# =====================================================
if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # ----- processing -----
    clip, gamma = estimate_params(img)

    wb  = white_balance(img)
    lab = lab_correction(wb, clip)
    vis = visibility_restore(lab, gamma)
    fused = fuse(lab, vis)

    inp = preprocess(fused)

    with torch.no_grad():
        out = model(inp)

    final = postprocess(out)

    # =================================================
    # CREATE TABS LIKE PAGES
    # =================================================
    t1, t2, t3, t4 = st.tabs([
        "🖼 Enhancement",
        "🔎 All Stages",
        "📊 Evaluation",
        "⬇ Download"
    ])

    # -------------------------------------------------
    with t1:
        st.subheader("Before vs After")

        c1, c2 = st.columns(2)

        with c1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")

        with c2:
            st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB),caption="Enhanced")

    # -------------------------------------------------
    with t2:
        st.subheader("Processing Stages")

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="1. Input")

        st.image(cv2.cvtColor(wb, cv2.COLOR_BGR2RGB), caption="2. White Balance")

        st.image(cv2.cvtColor(lab, cv2.COLOR_BGR2RGB), caption="3. LAB Correction")

        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="4. Visibility")

        st.image(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB), caption="5. Fusion")

        st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="6. Final ADGOL")

    # -------------------------------------------------
    with t3:
        st.subheader("Evaluation Metrics")

        psnr, ssim, uciqe, uiqm = compute_metrics(fused, final)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("PSNR", f"{psnr:.2f}")
        c2.metric("SSIM", f"{ssim:.3f}")
        c3.metric("UCIQE", f"{uciqe:.3f}")
        c4.metric("UIQM", f"{uiqm:.3f}")

        # stage wise
        st.subheader("Stage-Wise Perceptual Quality")

        def perceptual(x):
            return round(np.mean(x)/255,3), round(np.std(x)/255,3)

        stages = [img, wb, lab, vis, fused, final]
        names  = ["Input","WB","LAB","Visibility","Fusion","Final"]

        rows = []
        for n,s in zip(names, stages):
            u,q = perceptual(s)
            rows.append([n,u,q])

        st.table({
            "Stage":[r[0] for r in rows],
            "UCIQE":[r[1] for r in rows],
            "UIQM":[r[2] for r in rows]
        })

    # -------------------------------------------------
    # -------------------------------------------------
    with t4:
        st.subheader("Download Results")

        # ---- Individual Downloads ----
        st.markdown("### Download Individual Images")

        def to_bytes(im):
            return cv2.imencode(".png", im)[1].tobytes()

        st.download_button("Download Input Image",
                        to_bytes(img),
                        file_name="1_input.png",
                        mime="image/png")

        st.download_button("Download White Balance",
                        to_bytes(wb),
                        file_name="2_white_balance.png",
                        mime="image/png")

        st.download_button("Download LAB Corrected",
                        to_bytes(lab),
                        file_name="3_lab_corrected.png",
                        mime="image/png")

        st.download_button("Download Visibility Restored",
                        to_bytes(vis),
                        file_name="4_visibility.png",
                        mime="image/png")

        st.download_button("Download Fusion Output",
                        to_bytes(fused),
                        file_name="5_fusion.png",
                        mime="image/png")

        st.download_button("Download Final ADGOL Output",
                        to_bytes(final),
                        file_name="6_final_adgol.png",
                        mime="image/png")


        # ---- Download All as ZIP ----
        st.markdown("### Download All Images Together")

        import io, zipfile

        def create_zip():
            buf = io.BytesIO()

            with zipfile.ZipFile(buf, "w") as z:

                z.writestr("1_input.png", to_bytes(img))
                z.writestr("2_white_balance.png", to_bytes(wb))
                z.writestr("3_lab_corrected.png", to_bytes(lab))
                z.writestr("4_visibility.png", to_bytes(vis))
                z.writestr("5_fusion.png", to_bytes(fused))
                z.writestr("6_final_adgol.png", to_bytes(final))

            return buf.getvalue()


        st.download_button(
            "Download All Stages (ZIP)",
            create_zip(),
            file_name="underwater_results.zip",
            mime="application/zip"
        )
else:
    st.info("Please upload an image to begin.")
