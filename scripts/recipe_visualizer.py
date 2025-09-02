import sys
sys.path.append('../utils') 
from utils import *

import streamlit as st


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json




# loading model
path_folder = "../embeddings/"
name_embedding = "recipe_embeddings.npy"
name_index = "recipe_index.faiss"

@st.cache_resource
def load_model_and_indexes():
    return load_model(path_folder + name_embedding, path_folder + name_index)

model, embeddings, indexes = load_model_and_indexes()

# Page configuration
st.set_page_config(
    page_title="Recipe Visualizer",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sample recipe data
df = pd.read_pickle("../data/RAW_recipes.pkl")
df['name'] = df['name'].str.capitalize()

dict_to_show = {"name": st.column_config.TextColumn("Recipe Name", width="small"),
                "minutes": st.column_config.TextColumn("Time (mins)", width="small"),
                "n_ingredients": st.column_config.TextColumn("Number of ingredients", width="small")}

cols_to_show = list(dict_to_show.keys())

@st.cache_data
def cached_search(query, top_number):
    return search_recipe(query, model= model, indexes= indexes, df= df, top_number=top_number)

def main():
    style_heading = 'text-align: center'
    st.markdown(f"<h1 style='{style_heading}'>What do you want to cook?</h1>", unsafe_allow_html=True)
    search_query = st.text_input("Search for something:")
    
    
    if search_query.strip():
        if "last_query" not in st.session_state or st.session_state.last_query != search_query:
            st.session_state.results = cached_search(search_query, 10)
            st.session_state.last_query = search_query
            
        results = st.session_state.results
        
        # if its found then
        st.subheader(f"Search Results ({len(results)} found)")
                    
        st.write("**Select recipes from the table below to run algorithms:**")
        
        selected_rows = st.dataframe(
            results.loc[:, cols_to_show],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=dict_to_show
        )

        
        if selected_rows.selection.rows:
            index_selected = selected_rows.selection.rows[0]
            chosen_recipe = results.iloc[index_selected]            
            
            name = chosen_recipe.loc["name"].capitalize()
            minutes = chosen_recipe.loc["minutes"]
            tags = eval(chosen_recipe.loc["tags"])
            ingredients = eval(chosen_recipe.loc['ingredients'])
            steps = eval(chosen_recipe.loc["steps"])
            
            st.header(name)
            st.subheader(f"⏰ Preparation time: {minutes} minutes")
            
            # Recipe tags
            # st.write("**Tags:**", " | ".join([f"🏷️ {tag}" for tag in recipe['tags']]))
            # st.write("**Cuisine:**", f"🌍 {recipe['cuisine']}")
            
            # Ingredients section
            st.subheader("📋 Ingredients")
            
            
            # Display ingredients in a nice format
            for ingredient in ingredients:
                st.write(f"• {ingredient.capitalize()}")
            
            # Instructions section
            st.subheader("👩‍🍳 Instructions")
            for i, instruction in enumerate(steps, 1):
                st.write(f"**{i}.** {instruction.capitalize()}")
            
            # selected_recipe_names = [results.iloc[i]['Recipe Name'] for i in selected_rows.selection.rows]
            
            # st.success(f"✅ Selected {len(selected_recipe_names)} recipe(s): {', '.join(selected_recipe_names)}")
            
            # # Algorithm selection and execution section
            # st.subheader("🧮 Run Algorithms on Selected Recipes")
            
            # col1, col2 = st.columns([2, 1])
        
    else:
        st.subheader(f"")

# Additional utility functions
def calculate_nutrition_estimate(ingredients):
    """
    Placeholder function for nutritional calculation
    In a real app, this would integrate with a nutrition API
    """
    # This would use a nutrition database API like USDA or Edamam
    return {
        "calories": "Estimated 350 per serving",
        "protein": "15g",
        "carbs": "45g",
        "fat": "12g"
    }

# def export_recipe_json(recipe_data):
#     """Export recipe as JSON"""
#     return json.dumps(recipe_data, indent=2)

if __name__ == "__main__":
    main()









