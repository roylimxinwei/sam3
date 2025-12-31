import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import pandas as pd

from fastai.vision.all import load_learner, PILImage

import re
import web_scrape

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Multi-Model Food Classifier", page_icon="🍱", layout="wide")
st.title("🍱 Food Classifier Comparison (Multiple Models)")

# --- Utility ---
@st.cache_resource(show_spinner=True)
def load_model_from_bytes(pkl_bytes: bytes, name="model"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", prefix=name)
    tmp.write(pkl_bytes)
    tmp.flush()
    return load_learner(tmp.name)

@st.cache_resource(show_spinner=True)
def load_model_from_path(path: Path):
    return load_learner(path)

# --- Sidebar: Upload models ---
st.sidebar.header("📦 Models")
default_model = None
models = {}

# Optional default model from folder
if Path("food_classifier.pkl").exists():
    default_model = load_model_from_path(Path("food_classifier.pkl"))
    models["default_model"] = default_model
    st.sidebar.success("Loaded default: food_classifier.pkl")

# Upload multiple models
uploaded_models = st.sidebar.file_uploader("Upload one or more fastai models (.pkl)", type=["pkl"], accept_multiple_files=True)
if uploaded_models:
    for file in uploaded_models:
        name = Path(file.name).stem
        models[name] = load_model_from_bytes(file.read(), name=name)

if not models:
    st.info("⬆️ Upload at least one model to begin.")
    st.stop()

st.success(f"✅ {len(models)} model(s) loaded")

# --- Sidebar: Settings ---
st.sidebar.header("⚙️ Prediction Settings")
topk = st.sidebar.slider("Top-k probabilities", 1, 10, 5)
conf_digits = st.sidebar.slider("Decimal places", 2, 6, 4)

# --- Single Image Tab ---
tab_single, tab_batch = st.tabs(["Single Image", "Batch Images"])

with tab_single:
    up_img = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single_image")
    if up_img:
        img = Image.open(up_img).convert("RGB")
        st.image(img, caption="Uploaded Image", width='content')
        
        with st.spinner("Predicting with all models..."):
            col_list = st.columns(len(models))
            for (model_name, learner), col in zip(models.items(), col_list):
                pred_class, pred_idx, probs = learner.predict(PILImage.create(img))
                classes = learner.dls.vocab
                topk_idx = probs.argsort(descending=True)[:topk]
                top_labels = [classes[i] for i in topk_idx]
                top_probs = [round(float(probs[i]), conf_digits) for i in topk_idx]

                df = pd.DataFrame({
                    "Class": top_labels,
                    "Prob": top_probs
                })

                with col:
                    st.markdown(f"**🧠Model: {model_name}**")
                    st.markdown(f"**Prediction:** {pred_class}")
                    st.dataframe(df, width='stretch')
        
        # --- Nutrition Lookup Section ---
        st.divider()
        st.subheader("🥗 Nutrition Lookup")
        
        # Use the first model's prediction as the search term
        first_model_name = list(models.keys())[0]
        first_learner = models[first_model_name]
        pred_class, _, _ = first_learner.predict(PILImage.create(img))
        search_term = str(pred_class)
        
        # Store results in session state to persist across reruns
        if "nutrition_results" not in st.session_state:
            st.session_state.nutrition_results = None
            st.session_state.nutrition_search_term = None
        
        # Button to trigger the search
        if st.button("🔍 Search for nutrition info", key="search_nutrition"):
            with st.spinner(f"Searching for '{search_term}'..."):
                results = web_scrape.top5_scrape(search_term=search_term, headless=True)
                st.session_state.nutrition_results = results
                st.session_state.nutrition_search_term = search_term
        
        # Display results if we have them
        if st.session_state.nutrition_results is not None:
            results = st.session_state.nutrition_results
            
            if not results:
                st.warning("No results found.")
            else:
                st.info("Select one of the search results:")
                # results is now a list of strings
                options = [f"{i+1}. {name}" for i, name in enumerate(results)]

                # Radio returns the *label* the user picked
                choice_label = st.radio("Choose one:", options, index=0, key="nutrition_choice")

                # Map label back to index
                chosen_idx = options.index(choice_label)
                chosen_name = results[chosen_idx]

                if st.button("✅ Get Nutrition Info", key="confirm_nutrition"):
                    with st.spinner(f"Fetching nutrition for '{chosen_name}'..."):
                        nutrition_data = web_scrape.scrape_nutrition_by_index(
                            search_term=st.session_state.nutrition_search_term,
                            chosen_idx=chosen_idx,
                            headless=True
                        )
                        # result_data = {
                        #     "search_term": search_term,
                        #     "name": name,
                        #     "description": description,
                        #     "extra_info": extra_info,
                        #     "nutrition": nutrition
                        # }
                        if nutrition_data:
                            st.success(f"**{nutrition_data['name']}**")
                            st.write(f"*{nutrition_data['description']}*")
                            
                            # Display nutrition as a table
                            if nutrition_data['nutrition']:
                                nutrition_df = pd.DataFrame(
                                    list(nutrition_data['nutrition'].items()),
                                    columns=["Nutrient", "Per 100g"]
                                )
                                st.dataframe(nutrition_df, width='stretch')
                            default_size = nutrition_data["extra_info"].get("Default Serving Size")
                            st.write(f"Default Serving Size: {default_size}")
                            serving_grams = re.search(r"(\d+(?:\.\d+)?)\s*g", default_size)
                            if serving_grams:
                                grams = float(serving_grams.group(1))

                            # Convert all nutrient values (strings) → float safely
                            nutrition_per100g = {
                                nutrient: float(value) if value.replace('.', '', 1).isdigit() else 0.0
                                for nutrient, value in nutrition_data['nutrition'].items()
                            }

                            # Compute nutrition per serving
                            estimated_nutrition = {
                                nutrient: round(val * grams / 100, 2)
                                for nutrient, val in nutrition_per100g.items()
                            }

                            # # Show entire table
                            # est_nutrition_df = pd.DataFrame(
                            #     list(estimated_nutrition.items()),
                            #     columns=["Nutrient", "Estimated Amount (per serving)"]
                            # )

                            # Nutrients you want to keep
                            keep_nutrients = [
                                "Energy (kcal)",
                                "Protein (g)",
                                "Carbohydrate (g)",
                                "Total Fat (g)",
                                "Dietary Fibre (g)",
                            ]

                            # Filter dictionary
                            filtered_estimated = {
                                nutrient: amount
                                for nutrient, amount in estimated_nutrition.items()
                                if nutrient in keep_nutrients
                            }

                            # Create filtered DataFrame
                            est_nutrition_df = pd.DataFrame(
                                list(filtered_estimated.items()),
                                columns=["Nutrient", "Estimated Amount (per serving)"]
                            )

                            st.subheader("Estimated Nutritional Values per Default Serving Size")
                            st.dataframe(est_nutrition_df, width='stretch')


                        else:
                            st.error("Failed to fetch nutrition data.")


                


# --- Batch Image Tab ---
with tab_batch:
    ups = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_images")
    if ups:
        all_results = []
        prog = st.progress(0)
        for i, f in enumerate(ups, start=1):
            img_result = {"File": f.name}
            try:
                img = Image.open(f).convert("RGB")
                for model_name, learner in models.items():
                    pred_class, pred_idx, probs = learner.predict(PILImage.create(img))
                    img_result[f"{model_name}_prediction"] = str(pred_class)
                    img_result[f"{model_name}_conf"] = round(float(probs[pred_idx]), conf_digits)
            except Exception as e:
                for model_name in models:
                    img_result[f"{model_name}_prediction"] = "ERROR"
                    img_result[f"{model_name}_conf"] = 0.0
            all_results.append(img_result)
            prog.progress(i / len(ups))
        prog.empty()

        df = pd.DataFrame(all_results)
        st.dataframe(df, width='stretch')
        csv = df.to_csv(index=False).encode()
        st.download_button("Download results as CSV", csv, "multi_model_predictions.csv", "text/csv")

 