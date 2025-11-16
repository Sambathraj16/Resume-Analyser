import streamlit as st
from main import model, Agent
import os
import re
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Resume-Job Matcher",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .stTextArea textarea {
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def extract_score(text, pattern, default=0):
    """Extract numerical score from text using regex pattern"""
    try:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 10)  # Ensure score is between 0-10
        return default
    except:
        return default

def create_donut_chart(score, title, max_score=10):
    """Create a donut chart for score visualization with color coding"""
    # Determine color based on score
    if score >= 7:
        color = '#28a745'  # Green for good
        text_color = '#28a745'
    elif score >= 5:
        color = '#ffc107'  # Yellow for moderate
        text_color = '#f39c12'
    else:
        color = '#dc3545'  # Red for low
        text_color = '#dc3545'

    percentage = (score / max_score) * 100

    fig = go.Figure(data=[go.Pie(
        values=[score, max_score - score],
        hole=0.7,
        marker=dict(colors=[color, '#e9ecef']),
        textinfo='none',
        hoverinfo='skip',
        showlegend=False
    )])

    fig.update_layout(
        annotations=[
            dict(
                text=f'<b>{score:.1f}</b><br><span style="font-size:14px">out of {max_score}</span>',
                x=0.5, y=0.5,
                font=dict(size=28, color=text_color),
                showarrow=False
            )
        ],
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16, color='#2c3e50')),
        height=250,
        margin=dict(t=50, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def parse_list_from_text(text, keywords):
    """Extract list items from text based on keywords"""
    items = []
    lines = text.split('\n')
    capture = False

    for line in lines:
        line = line.strip()
        # Check if line contains any keyword
        if any(keyword.lower() in line.lower() for keyword in keywords):
            capture = True
            continue
        # Check for list items (bullets, numbers, dashes)
        if capture and (line.startswith('-') or line.startswith('•') or
                       line.startswith('*') or re.match(r'^\d+\.', line)):
            items.append(re.sub(r'^[-•*\d.)\s]+', '', line).strip())
        elif capture and line and not line.startswith('-'):
            # Stop capturing if we hit a non-list line
            if items:  # Only stop if we've already captured something
                break

    return items

# Initialize session state
if 'resume_saved' not in st.session_state:
    st.session_state.resume_saved = False
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "llama-3.1-70b-versatile"

# Header
st.markdown('<div class="main-header">Resume-Job Description Matcher</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key input - moved to top
    st.subheader("🔑 API Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your API key from https://console.groq.com",
        placeholder="Enter your Groq API key here"
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("✓ API Key configured!")
    else:
        st.warning("⚠️ Please enter API key to proceed")

    st.markdown("---")

    # Model selection - enhanced
    st.subheader("🤖 Select LLM Model")
    model_options = {
        "Llama 3.1 8B (Recommended)": "llama-3.1-8b-instant",
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Meta 4-12B": "meta-llama/llama-guard-4-12b",
        "OpenAI GPT OSS": "openai/gpt-oss-safeguard-20b"
    }

    selected_model_name = st.selectbox(
        "Choose your preferred model",
        list(model_options.keys()),
        index=0,
        help="Different models may provide varying levels of analysis depth and speed"
    )
    st.session_state.llm_model = model_options[selected_model_name]

    # Display model info
    st.info(f"**Active Model:** {st.session_state.llm_model}")

    st.markdown("---")

    # Resume section in sidebar
    st.subheader("📄 Your Resume")

    if not st.session_state.resume_saved:
        resume_input = st.text_area(
            "Paste your resume here",
            height=200,
            placeholder="Paste your complete resume text here...",
            help="This will be saved and used for all job description comparisons",
            key="sidebar_resume"
        )

        if st.button("💾 Save Resume", type="primary", use_container_width=True):
            if resume_input.strip():
                st.session_state.resume_text = resume_input
                st.session_state.resume_saved = True
                st.rerun()
            else:
                st.error("Please enter your resume!")
    else:
        st.success("✓ Resume saved successfully!")

        with st.expander("👁️ View/Edit Resume"):
            resume_edit = st.text_area(
                "Your Resume",
                value=st.session_state.resume_text,
                height=200,
                key="resume_view"
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("💾 Update", type="primary", use_container_width=True):
                    st.session_state.resume_text = resume_edit
                    st.success("Resume updated!")
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                    st.session_state.resume_saved = False
                    st.rerun()

    st.markdown("---")

    # Instructions
    st.subheader("📖 How to use")
    st.markdown("""
    1. Enter your Groq API key
    2. Select your preferred LLM model
    3. Paste your resume and save it
    4. Paste job descriptions in main area
    5. Click 'Analyze Match' to get insights
    6. Review detailed analysis in tabs
    """)

    st.markdown("---")

    # Clear history button
    if st.button("🗑️ Clear Analysis History", type="secondary", use_container_width=True):
        st.session_state.analysis_history = []
        st.rerun()

# Main content area
st.markdown('<div class="section-header">📋 Job Description Analysis</div>', unsafe_allow_html=True)

if not st.session_state.resume_saved:
    st.info("👈 Please save your resume in the sidebar first before analyzing job descriptions.")
    st.stop()

job_description = st.text_area(
    "Paste the job description you want to analyze",
    height=350,
    placeholder="Paste the complete job description here...\n\nInclude:\n- Job title and company\n- Required skills and qualifications\n- Responsibilities\n- Experience requirements\n- Any other relevant details",
    help="Paste one job description at a time for detailed analysis"
)

analyze_button = st.button("🔍 Analyze Match", type="primary", disabled=not job_description.strip(), use_container_width=False)

# Analysis section
if st.session_state.resume_saved and 'analyze_button' in locals() and analyze_button:
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar!")
    else:
        with st.spinner("Analyzing your resume against the job description... This may take a moment."):
            try:
                # Prepare input state
                input_state = {
                    'llm_model': st.session_state.llm_model,
                    'raw_resume': st.session_state.resume_text,
                    'raw_job_description': job_description
                }

                # Run the workflow
                result = model.invoke(input_state)

                # Store in history
                st.session_state.analysis_history.insert(0, {
                    'job_description': job_description[:200] + "..." if len(job_description) > 200 else job_description,
                    'result': result
                })

                st.success("Analysis completed successfully!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)

# Display analysis results
if st.session_state.analysis_history:
    st.markdown("---")
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    for idx, analysis in enumerate(st.session_state.analysis_history):
        with st.expander(f"Analysis #{len(st.session_state.analysis_history) - idx} - {analysis['job_description'][:100]}...", expanded=(idx == 0)):
            result = analysis['result']

            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Overview & Scores",
                "🎯 Key Skills Required",
                "💪 Your Strengths",
                "⚠️ Your Weaknesses",
                "📝 Resume Modifications",
                "🔧 Improvement Areas",
                "📈 Detailed Analysis",
                "📄 Raw Data"
            ])

            with tab1:
                st.markdown("### 📊 Match Overview & Scores")

                if 'final_decision' in result:
                    decision_text = result['final_decision'].content if hasattr(result['final_decision'], 'content') else str(result['final_decision'])

                    # Extract scores
                    fit_score = extract_score(decision_text, r'(?:fit|rating|score).*?(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)', default=5)
                    overqualification_score = extract_score(decision_text, r'overqualification.*?(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)', default=0)

                    # Display score charts
                    col_score1, col_score2 = st.columns(2)

                    with col_score1:
                        st.plotly_chart(
                            create_donut_chart(fit_score, "🎯 Job Fit Score"),
                            use_container_width=True
                        )

                    with col_score2:
                        st.plotly_chart(
                            create_donut_chart(overqualification_score, "⚡ Overqualification Score"),
                            use_container_width=True
                        )

                    st.markdown("---")

                    # Display full decision
                    st.markdown("### 📋 Final Recommendation")
                    st.markdown(decision_text)

                else:
                    st.info("Final decision not available")

            with tab2:
                st.markdown("### 🎯 Key Skills Required by Employer")

                if 'parsed_job_description' in result:
                    parsed_jd = result['parsed_job_description']

                    # Programming Languages
                    st.markdown("#### 💻 Programming Languages")
                    if parsed_jd.get('programming_languages'):
                        cols = st.columns(3)
                        for idx, lang in enumerate(parsed_jd['programming_languages']):
                            with cols[idx % 3]:
                                st.markdown(f"**{lang}**")
                    else:
                        st.info("No specific programming languages mentioned")

                    st.markdown("---")

                    # Technical Skills
                    st.markdown("#### 🔧 Technical Skills Required")
                    if parsed_jd.get('technical_skills'):
                        cols = st.columns(2)
                        for idx, skill in enumerate(parsed_jd['technical_skills']):
                            with cols[idx % 2]:
                                st.markdown(f"✓ {skill}")
                    else:
                        st.info("No specific technical skills mentioned")

                    st.markdown("---")

                    # Tools and Technologies
                    st.markdown("#### 🛠️ Tools & Technologies")
                    if parsed_jd.get('tools_and_technologies'):
                        cols = st.columns(3)
                        for idx, tool in enumerate(parsed_jd['tools_and_technologies']):
                            with cols[idx % 3]:
                                st.markdown(f"⚙️ {tool}")
                    else:
                        st.info("No specific tools mentioned")

                    st.markdown("---")

                    # Soft Skills
                    st.markdown("#### 🤝 Soft Skills Expected")
                    if parsed_jd.get('soft_skills'):
                        cols = st.columns(2)
                        for idx, skill in enumerate(parsed_jd['soft_skills']):
                            with cols[idx % 2]:
                                st.markdown(f"• {skill}")
                    else:
                        st.info("No specific soft skills mentioned")

                    st.markdown("---")

                    # Experience and Education
                    col_req1, col_req2 = st.columns(2)
                    with col_req1:
                        st.markdown("#### 📚 Education Required")
                        st.info(parsed_jd.get('education', 'Not specified'))

                    with col_req2:
                        st.markdown("#### ⏱️ Experience Required")
                        st.info(parsed_jd.get('years_of_experience', 'Not specified'))

                else:
                    st.warning("Job description data not available")

            with tab3:
                st.markdown("### 💪 Your Strengths")

                strengths_list = []

                # Get from parsed resume
                if 'parsed_resume' in result and result['parsed_resume'].get('strengths'):
                    strengths_text = result['parsed_resume']['strengths']
                    st.markdown("#### 🌟 Identified Strengths")
                    st.success(strengths_text)

                # Get from final decision
                if 'final_decision' in result:
                    decision_text = result['final_decision'].content if hasattr(result['final_decision'], 'content') else str(result['final_decision'])
                    strengths_from_decision = parse_list_from_text(decision_text, ['strength', 'strong', 'advantage'])

                    if strengths_from_decision:
                        st.markdown("---")
                        st.markdown("#### ✨ Key Advantages for This Role")
                        for strength in strengths_from_decision:
                            st.markdown(f"✅ **{strength}**")

                if not strengths_list and 'parsed_resume' not in result:
                    st.info("No specific strengths identified")

            with tab4:
                st.markdown("### ⚠️ Your Weaknesses & Areas of Concern")

                # Get from parsed resume
                if 'parsed_resume' in result and result['parsed_resume'].get('weaknesses'):
                    weaknesses_text = result['parsed_resume']['weaknesses']
                    st.markdown("#### 🔍 General Weaknesses Identified")
                    st.warning(weaknesses_text)

                # Get from final decision
                if 'final_decision' in result:
                    decision_text = result['final_decision'].content if hasattr(result['final_decision'], 'content') else str(result['final_decision'])
                    weaknesses_from_decision = parse_list_from_text(decision_text, ['weakness', 'weak', 'gap', 'concern', 'lack', 'missing'])

                    if weaknesses_from_decision:
                        st.markdown("---")
                        st.markdown("#### 🎯 Specific Gaps for This Role")
                        for weakness in weaknesses_from_decision:
                            st.markdown(f"⚠️ **{weakness}**")

                if 'parsed_resume' not in result or not result['parsed_resume'].get('weaknesses'):
                    if not weaknesses_from_decision:
                        st.success("✓ No significant weaknesses identified!")

            with tab5:
                st.markdown("### 📝 Resume Modification Suggestions")

                if 'final_decision' in result:
                    decision_text = result['final_decision'].content if hasattr(result['final_decision'], 'content') else str(result['final_decision'])

                    # Look for resume enhancement/modification suggestions
                    modifications = parse_list_from_text(
                        decision_text,
                        ['resume', 'modify', 'add', 'highlight', 'emphasize', 'include', 'showcase']
                    )

                    if modifications:
                        st.markdown("#### ✏️ Recommended Changes to Your Resume")
                        for i, mod in enumerate(modifications, 1):
                            st.markdown(f"**{i}.** {mod}")
                    else:
                        # Try to extract any modification-related content
                        lines = decision_text.split('\n')
                        mod_section = []
                        capture = False

                        for line in lines:
                            if any(keyword in line.lower() for keyword in ['resume', 'cv', 'modify', 'enhance your', 'update your']):
                                capture = True
                            if capture and line.strip():
                                mod_section.append(line)
                            if capture and len(mod_section) > 5:
                                break

                        if mod_section:
                            st.markdown("#### ✏️ Resume Enhancement Recommendations")
                            st.markdown('\n'.join(mod_section))
                        else:
                            st.info("No specific resume modifications suggested")

                    st.markdown("---")

                    # Skills to highlight
                    st.markdown("#### 🌟 Skills to Highlight More Prominently")
                    if 'parsed_resume' in result and 'parsed_job_description' in result:
                        resume_skills = set(result['parsed_resume'].get('technical_skills', []))
                        jd_skills = set(result['parsed_job_description'].get('technical_skills', []))

                        matching_skills = resume_skills.intersection(jd_skills)

                        if matching_skills:
                            st.success("These skills match the job requirements - make sure they're prominent in your resume:")
                            cols = st.columns(2)
                            for idx, skill in enumerate(matching_skills):
                                with cols[idx % 2]:
                                    st.markdown(f"⭐ **{skill}**")
                        else:
                            st.info("Consider adding relevant skills mentioned in the job description")

                    st.markdown("---")

                    # Projects to emphasize
                    st.markdown("#### 💼 Experience/Projects to Emphasize")
                    if 'skill_assesment' in result:
                        skill_text = result['skill_assesment'].content if hasattr(result['skill_assesment'], 'content') else str(result['skill_assesment'])
                        relevant_exp = parse_list_from_text(skill_text, ['experience', 'project', 'emphasize', 'highlight', 'relevant'])

                        if relevant_exp:
                            for exp in relevant_exp:
                                st.markdown(f"📌 {exp}")
                        else:
                            st.info("Review your projects and emphasize those most relevant to this role")

                else:
                    st.warning("Analysis data not available")

            with tab6:
                st.markdown("### 🔧 Recommended Improvement Areas")

                if 'final_decision' in result:
                    decision_text = result['final_decision'].content if hasattr(result['final_decision'], 'content') else str(result['final_decision'])

                    # Extract improvement recommendations
                    improvements = parse_list_from_text(
                        decision_text,
                        ['recommendation', 'improve', 'enhance', 'focus', 'develop', 'should', 'consider', 'work on']
                    )

                    if improvements:
                        st.markdown("#### 🎯 Actionable Steps to Improve")
                        for i, improvement in enumerate(improvements, 1):
                            st.markdown(f"**{i}.** {improvement}")
                    else:
                        # If no specific list, try to extract recommendation sections
                        lines = decision_text.split('\n')
                        in_recommendation_section = False
                        recommendation_text = []

                        for line in lines:
                            if any(keyword in line.lower() for keyword in ['recommendation', 'improve', 'enhance', 'focus on']):
                                in_recommendation_section = True
                            if in_recommendation_section:
                                recommendation_text.append(line)

                        if recommendation_text:
                            st.markdown('\n'.join(recommendation_text))
                        else:
                            st.info("No specific improvement recommendations provided")

                # Skills gap from skill assessment
                if 'skill_assesment' in result:
                    st.markdown("---")
                    st.markdown("#### 📚 Skills to Develop")
                    skill_text = result['skill_assesment'].content if hasattr(result['skill_assesment'], 'content') else str(result['skill_assesment'])
                    skill_gaps = parse_list_from_text(skill_text, ['gap', 'missing', 'need', 'lack', 'should learn', 'acquire'])

                    if skill_gaps:
                        cols = st.columns(2)
                        for idx, gap in enumerate(skill_gaps):
                            with cols[idx % 2]:
                                st.markdown(f"📖 {gap}")

            with tab7:
                st.markdown("### 📈 Detailed Analysis")

                # Skills Assessment
                st.markdown("#### 🔧 Skills Assessment")
                if 'skill_assesment' in result:
                    skill_text = result['skill_assesment'].content if hasattr(result['skill_assesment'], 'content') else str(result['skill_assesment'])
                    with st.expander("View Full Skills Assessment", expanded=False):
                        st.write(skill_text)
                else:
                    st.info("Skills assessment not available")

                st.markdown("---")

                # Experience Assessment
                st.markdown("#### 💼 Experience Assessment")
                if 'experience_assesment' in result:
                    exp_text = result['experience_assesment'].content if hasattr(result['experience_assesment'], 'content') else str(result['experience_assesment'])
                    with st.expander("View Full Experience Assessment", expanded=False):
                        st.write(exp_text)
                else:
                    st.info("Experience assessment not available")

            with tab8:
                st.markdown("### 📄 Parsed Data")

                # Parsed Resume
                st.markdown("#### Parsed Resume Information")
                if 'parsed_resume' in result:
                    parsed_resume = result['parsed_resume']

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Basic Information**")
                        st.write(f"**Name:** {parsed_resume.get('candidate_name', 'N/A')}")
                        st.write(f"**Job Title:** {parsed_resume.get('job_title', 'N/A')}")
                        st.write(f"**Industry:** {parsed_resume.get('industry', 'N/A')}")
                        st.write(f"**Experience:** {parsed_resume.get('years_of_experience', 'N/A')}")
                        st.write(f"**Education:** {parsed_resume.get('education', 'N/A')}")
                        st.write(f"**Location:** {parsed_resume.get('work_location', 'N/A')}")

                    with col_b:
                        st.markdown("**Project Information**")
                        st.write(f"**Number of Projects:** {parsed_resume.get('no_of_projects', 'N/A')}")
                        st.write(f"**Recent Project:** {parsed_resume.get('recent_project_title', 'N/A')}")

                    st.markdown("**Skills**")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.write("**Programming Languages:**")
                        if parsed_resume.get('programming_languages'):
                            for lang in parsed_resume['programming_languages']:
                                st.write(f"- {lang}")

                    with col_d:
                        st.write("**Technical Skills:**")
                        if parsed_resume.get('technical_skills'):
                            for skill in parsed_resume['technical_skills']:
                                st.write(f"- {skill}")
                else:
                    st.info("Parsed resume not available")

                st.markdown("---")

                # Parsed Job Description
                st.markdown("#### Parsed Job Description")
                if 'parsed_job_description' in result:
                    parsed_jd = result['parsed_job_description']

                    col_g, col_h = st.columns(2)
                    with col_g:
                        st.markdown("**Job Details**")
                        st.write(f"**Company:** {parsed_jd.get('company_name', 'N/A')}")
                        st.write(f"**Job Title:** {parsed_jd.get('job_title', 'N/A')}")
                        st.write(f"**Location:** {parsed_jd.get('job_location', 'N/A')}")
                        st.write(f"**Industry:** {parsed_jd.get('industry', 'N/A')}")
                        st.write(f"**Required Experience:** {parsed_jd.get('years_of_experience', 'N/A')}")
                        st.write(f"**Education:** {parsed_jd.get('education', 'N/A')}")

                    with col_h:
                        st.markdown("**Required Skills**")
                        st.write("**Programming Languages:**")
                        if parsed_jd.get('programming_languages'):
                            for lang in parsed_jd['programming_languages']:
                                st.write(f"- {lang}")

                        st.write("**Technical Skills:**")
                        if parsed_jd.get('technical_skills'):
                            for skill in parsed_jd['technical_skills']:
                                st.write(f"- {skill}")

                    st.markdown("**Responsibilities**")
                    st.write(parsed_jd.get('responsibilities', 'N/A'))
                else:
                    st.info("Parsed job description not available")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #888; font-size: 0.9rem;">'
    'Resume-Job Description Matcher | Powered by LangGraph & Groq'
    '</div>',
    unsafe_allow_html=True
)
