import streamlit as st
import SessionState
from prompts import PROMPT_LIST
from wit_index import WitIndex
import random
import time

# st.set_page_config(page_title="Image Search")

# vector_length = 128
wit_index_path = f"./models/wit_faiss.idx"
model_name = f"./models/distilbert-base-wit"
wit_dataset_path = "./models/wit_dataset.pkl"


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_wit_index():
    st.write("Loading the WIT index, dataset and the DistillBERT model..")
    wit_index = WitIndex(wit_index_path, model_name, wit_dataset_path, gpu=False)
    return wit_index

# st.cache is disabled temporarily because the inference could take forever using newer streamlit version.
# @st.cache(suppress_st_warning=True)
def process(text: str, top_k: int = 10):
    # st.write("Cache miss: process")
    distance, index, image_info = wit_index.search(text, top_k=top_k)
    return distance, index, image_info


st.title("Image Search")

st.markdown(
    """
    This application is a demo for sentence-based image search using 
    [WIT dataset](https://github.com/google-research-datasets/wit). We use DistillBert to encode the sentences 
    and Facebook's Faiss to search the vector embeddings.
    """
)
session_state = SessionState.get(prompt=None, prompt_box=None, text=None)
ALL_PROMPTS = list(PROMPT_LIST.keys())+["Custom"]
prompt = st.selectbox('Prompt', ALL_PROMPTS, index=len(ALL_PROMPTS)-1)
# Update prompt
if session_state.prompt is None:
    session_state.prompt = prompt
elif session_state.prompt is not None and (prompt != session_state.prompt):
    session_state.prompt = prompt
    session_state.prompt_box = None
    session_state.text = None
else:
    session_state.prompt = prompt

# Update prompt box
if session_state.prompt == "Custom":
    session_state.prompt_box = "Enter your text here"
else:
    if session_state.prompt is not None and session_state.prompt_box is None:
        session_state.prompt_box = random.choice(PROMPT_LIST[session_state.prompt])

session_state.text = st.text_area("Enter text", session_state.prompt_box)

top_k = st.sidebar.number_input(
    "Top k",
    value=6,
    min_value=1,
    max_value=10
)

wit_index = get_wit_index()
if st.button("Run"):
    with st.spinner(text="Getting results..."):
        st.subheader("Result")
        time_start = time.time()
        distances, index, image_info = process(text=session_state.text, top_k=int(top_k))
        time_end = time.time()
        time_diff = time_end-time_start
        print(f"Search in {time_diff} seconds")
        st.markdown(f"*Search in {time_diff:.5f} seconds*")
        for i, distance in enumerate(distances):
            try:
                st.image(image_info[i][0].replace("http:", "https:"), width=400)
            except FileNotFoundError:
                st.write(f"{image_info[i][0]} can't be displayed")
            st.write(f"{image_info[i][1]}. (D: {distance:.2f})")

        # Reset state
        session_state.prompt = None
        session_state.prompt_box = None
        session_state.text = None
