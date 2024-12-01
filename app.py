import os
import shutil
import zipfile
import streamlit as st  # Import Streamlit first

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="Codebase RAG Viewer & LLM Assistant", layout="wide")

from dotenv import load_dotenv
from codebase_rag_completed import get_main_files_content, perform_rag_without_streamlit, index_documents

# Load environment variables
load_dotenv()

# Set up Streamlit app
st.title("Codebase RAG Viewer & LLM Assistant")

# Option to choose input method
input_method = st.radio(
    "Choose input method:",
    ("Upload ZIP file", "Upload multiple files", "Provide local repository path")
)

# Initialize a session state to hold the codebase content
if 'codebase_content' not in st.session_state:
    st.session_state.codebase_content = []

# Function to process and display files
def process_and_display_files(file_contents):
    if file_contents:
        st.write(f"Found {len(file_contents)} files:")
        for file_info in file_contents:
            with st.expander(f"View {file_info['name']}"):
                st.text_area("Content", file_info['content'], height=200)
    else:
        st.warning("No supported files found.")

if input_method == "Upload ZIP file":
    # File uploader for ZIP file with a unique key
    uploaded_file = st.file_uploader("Upload a ZIP file here", type=["zip"], key="zip_upload_key")

    if uploaded_file is not None:
        with st.spinner("Processing ZIP file, please wait..."):
            st.write("File received successfully.")
            try:
                # Save the uploaded file locally
                temp_zip_path = "uploaded_codebase.zip"
                with open(temp_zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write("File saved successfully on the server.")

                # Extract ZIP contents to a temporary directory
                temp_dir = "uploaded_codebase"
                os.makedirs(temp_dir, exist_ok=True)

                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        # Skip node_modules to avoid issues
                        if 'node_modules' not in member:
                            zip_ref.extract(member, temp_dir)
                st.write("ZIP file extracted successfully, excluding 'node_modules'.")

                # Process the extracted files using `get_main_files_content`
                st.write("Analyzing codebase...")
                codebase_content = get_main_files_content(temp_dir)
                st.session_state.codebase_content = codebase_content  # Store in session state
                process_and_display_files(codebase_content)

                # Index the documents into Pinecone
                st.write("Indexing documents into Pinecone...")
                index_documents(codebase_content)
                st.success("Documents indexed successfully.")

            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP file. Please try uploading a valid ZIP archive.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
            finally:
                # Clean up the temporary ZIP file and directory using shutil
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception as e:
                        st.error(f"Failed to clean up temporary files: {e}")

elif input_method == "Upload multiple files":
    # Allow multiple file uploads
    uploaded_files = st.file_uploader(
        "Upload multiple files here", accept_multiple_files=True,
        type=["py", "js", "tsx", "jsx", "ipynb", "java", "cpp", "ts", "go", "rs", "vue", "swift", "c", "h"],
        key="multiple_files_upload"
    )

    if uploaded_files:
        with st.spinner("Processing files, please wait..."):
            st.write(f"{len(uploaded_files)} files uploaded successfully.")
            temp_dir = "uploaded_multiple_files"
            os.makedirs(temp_dir, exist_ok=True)
            try:
                file_contents = []
                for uploaded_file in uploaded_files:
                    # Save each file temporarily
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Get file content for analysis
                    file_content = {
                        "name": uploaded_file.name,
                        "content": uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    }
                    file_contents.append(file_content)

                # Process and display the files
                st.session_state.codebase_content = file_contents  # Store in session state
                process_and_display_files(file_contents)

                # Index the documents into Pinecone
                st.write("Indexing documents into Pinecone...")
                index_documents(file_contents)
                st.success("Documents indexed successfully.")

            except Exception as e:
                st.error(f"An unexpected error occurred while processing the files: {e}")
            finally:
                # Clean up the temporary files using shutil
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception as e:
                        st.error(f"Failed to clean up temporary files: {e}")

elif input_method == "Provide local repository path":
    # User input: Path to the repository
    repo_path = st.text_input("Enter the path to your local repository:", key="unique_repo_path")

    # Button to fetch file contents
    if st.button("Fetch Files", key="unique_fetch_files_button"):
        with st.spinner("Fetching files from the repository, please wait..."):
            if repo_path:
                try:
                    # Get file contents
                    files_content = get_main_files_content(repo_path)
                    st.session_state.codebase_content = files_content  # Store in session state
                    process_and_display_files(files_content)

                    # Index the documents into Pinecone
                    st.write("Indexing documents into Pinecone...")
                    index_documents(files_content)
                    st.success("Documents indexed successfully.")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a valid repository path.")

# Horizontal line to separate sections
st.markdown("---")

# Option to ask questions
st.header("Ask a Question About the Codebase")

# Creating a form to handle query submissions
with st.form(key="query_form"):
    query = st.text_input("Enter your question:", key="query_input_key")
    submit_button = st.form_submit_button("Submit Query")

# Handling the query
if submit_button:
    if query:
        with st.spinner("Processing your query, please wait..."):
            try:
                if not st.session_state.codebase_content:
                    st.error("Please upload a codebase first.")
                else:
                    response = perform_rag_without_streamlit(query)
                    st.subheader("Assistant's Response:")
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")
