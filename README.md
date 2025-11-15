# Resume-Job Description Matcher

An AI-powered application that analyzes your resume against job descriptions to provide insights, skill assessments, and recommendations.

## Features

- **One-time Resume Input**: Save your resume once and analyze multiple job descriptions
- **Comprehensive Analysis**: Get detailed insights on:
  - Skills assessment (technical and soft skills matching)
  - Experience evaluation
  - Strengths and weaknesses analysis
  - Overall fit rating (1-10 scale)
  - Recommendations for improvement
- **Multiple LLM Models**: Choose from various Groq models
- **Analysis History**: Keep track of all your job description analyses
- **Beautiful UI**: Clean, intuitive Streamlit interface

## Prerequisites

- Python 3.8 or higher
- Groq API Key (Get it from [https://console.groq.com](https://console.groq.com))

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Start the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

2. **Configure the app**:
   - Enter your Groq API key in the sidebar
   - Select your preferred LLM model (default: llama-3.1-70b-versatile)

3. **Add your resume**:
   - Paste your complete resume in the left text area
   - Click "Save Resume"

4. **Analyze job descriptions**:
   - Paste a job description in the right text area
   - Click "Analyze Match"
   - Wait for the analysis to complete

5. **View results**:
   - The analysis will appear below with multiple tabs:
     - **Final Decision**: Overall recommendation and fit rating
     - **Parsed Resume**: Structured information extracted from your resume
     - **Parsed Job Description**: Requirements and details from the job posting
     - **Skills Assessment**: Detailed skills matching analysis
     - **Experience Assessment**: Experience level evaluation

6. **Analyze more jobs**:
   - Simply paste another job description and click "Analyze Match" again
   - All analyses are saved in your session history

## Features Breakdown

### Skills Assessment
- Semantic skill matching (e.g., React ↔ JavaScript)
- Identification of transferable skills
- Skill gap analysis
- Recommendations for leveraging strengths

### Experience Assessment
- Experience level appropriateness
- Overqualification detection
- Career growth potential analysis
- Red flag identification in job descriptions

### Decision Making
- Aggregated insights from all assessments
- Fit rating (1-10 scale)
- Personalized recommendations
- Resume enhancement suggestions
- Clear guidance on whether to apply

## Tips for Best Results

1. **Resume Quality**: Provide a detailed, well-structured resume
2. **Complete Information**: Include all relevant skills, projects, and experience
3. **Full Job Description**: Paste the complete job posting for accurate analysis
4. **Multiple Analyses**: Compare multiple job descriptions to find the best fit

## Troubleshooting

- **API Key Error**: Make sure you've entered a valid Groq API key
- **Analysis Failed**: Check your internet connection and API key validity
- **Slow Performance**: Large resumes/job descriptions may take longer to process

## Project Structure

```
.
├── main.py                 # Core LangGraph workflow and agents
├── streamlit_app.py        # Streamlit UI application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technologies Used

- **Streamlit**: Web application framework
- **LangGraph**: Agent orchestration and workflow management
- **LangChain**: LLM integration
- **Groq**: Fast LLM inference
- **Pydantic**: Data validation and structured outputs

## License

This project is for personal use.
