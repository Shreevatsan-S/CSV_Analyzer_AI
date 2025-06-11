import os
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
load_dotenv()

# Load Gemini API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise Exception("GOOGLE_API_KEY not set in environment variables. Please configure it.")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)

# Define Prompt Template (modified to include user_prompt)
prompt_template = PromptTemplate(
    input_variables=["data_profile", "user_prompt"],
    template="""
You are an expert data scientist.

You are given the following data profile:

{data_profile}

The user has asked you to analyze this dataset with the following goal:

"{user_prompt}"

Based on the above goal, provide a brief inference or analysis of the dataset.
Optionally suggest key variables or trends the user should focus on.

Format your output clearly using markdown.
"""
)

# Simple data profiling function
def profile_dataframe(df: pd.DataFrame) -> str:
    import tabulate

    # Basic info
    num_rows = len(df)
    num_columns = len(df.columns)

    # Column names and data types
    col_info = []
    for col in df.columns:
        col_info.append([col, str(df[col].dtype), df[col].isnull().sum()])

    # Format as markdown table using tabulate
    col_table = tabulate.tabulate(
        col_info,
        headers=["Column Name", "Data Type", "Missing Values"],
        tablefmt="github"
    )

    # Build final profile string
    profile_str = f"""
**Number of rows:** {num_rows}  
**Number of columns:** {num_columns}  

### Column Details:

{col_table}
"""

    return profile_str


# Node 1: Profile Data
def node_profile_data(state):
    df = state["df"]
    data_profile = profile_dataframe(df)
    state["data_profile"] = data_profile
    return state

# Node 2: AI Suggest Visualization Ideas
def node_ai_suggest_viz(state):
    data_profile = state["data_profile"]
    user_prompt = state.get("user_prompt", "Provide general insights about the dataset.")
    prompt = prompt_template.format(data_profile=data_profile, user_prompt=user_prompt)
    output_parser = StrOutputParser()
    response = llm.invoke(prompt)
    parsed_response = output_parser.invoke(response)
    state["viz_ideas"] = parsed_response
    return state

# Node 3: Auto-Plot
def node_auto_plot(state):
    df = state["df"]
    user_prompt = state.get("user_prompt", "").lower()
    plot_path = None

    try:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_columns) == 0:
            print("No numeric columns found for plotting.")
            return state

        fig, ax = plt.subplots(figsize=(8, 6))

        # Simple heuristic-based plot selection based on prompt keywords
        if "trend" in user_prompt or "time" in user_prompt or "over time" in user_prompt:
            # Line plot
            x_col = df.columns[0]
            y_col = numeric_columns[0]
            ax.plot(df[x_col], df[y_col], marker='o', color='green')
            ax.set_title(f"Trend of {y_col} over {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif "correlation" in user_prompt or "relationship" in user_prompt or "scatter" in user_prompt:
            # Scatter plot
            if len(numeric_columns) >= 2:
                x_col = numeric_columns[0]
                y_col = numeric_columns[1]
                ax.scatter(df[x_col], df[y_col], alpha=0.6, color='purple')
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            else:
                print("Not enough numeric columns for scatter plot.")
                return state

        elif "distribution" in user_prompt or "histogram" in user_prompt:
            # Histogram
            col = numeric_columns[0]
            ax.hist(df[col], bins=20, color='blue', alpha=0.7)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        else:
            # Default fallback: simple bar plot of first numeric column
            col = numeric_columns[0]
            ax.bar(df.index, df[col], color='orange')
            ax.set_title(f"Bar Plot of {col} by Index")
            ax.set_xlabel("Index")
            ax.set_ylabel(col)

        plt.grid(True)

        # Save to temp file
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile.name)
        plt.close(fig)
        plot_path = tmpfile.name

    except Exception as e:
        print("Auto-Plot Error:", str(e))

    state["auto_plot_path"] = plot_path
    return state

# Build LangGraph
def build_agent_graph():
    workflow = StateGraph(state_schema=dict)
    workflow.add_node("ProfileData", node_profile_data)
    workflow.add_node("AISuggestViz", node_ai_suggest_viz)
    workflow.add_node("AutoPlot", node_auto_plot)

    # Edges
    workflow.set_entry_point("ProfileData")
    workflow.add_edge("ProfileData", "AISuggestViz")
    workflow.add_edge("AISuggestViz", "AutoPlot")
    workflow.add_edge("AutoPlot", END)

    return workflow.compile()

# Main analysis function 
def analyze_data_with_agent(df: pd.DataFrame, user_prompt: str):
    workflow = build_agent_graph()
    initial_state = {"df": df, "user_prompt": user_prompt}
    final_state = workflow.invoke(initial_state)

    profile_summary = final_state["data_profile"]
    viz_ideas = final_state["viz_ideas"]
    plot_path = final_state.get("auto_plot_path")

    return profile_summary, viz_ideas, plot_path
