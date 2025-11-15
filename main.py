from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from typing import TypedDict

# Define the Agent state structure
class Agent(TypedDict):
    llm_model: str
    raw_resume: str
    raw_job_description: str
    parsed_resume: dict
    parsed_job_description: dict
    skill_assesment: str
    experience_assesment: str
    final_decision: str

class ParseResume(BaseModel):
  job_title:str=Field(description="current job title employee searching ")
  years_of_experience:str= Field(description="total years of relevant experience")
  industry:str=Field(description="current industry which employee working")
  no_of_projects:str=Field(description="total no of projects employee given in resume")
  technical_skills:List=Field(description="Technical Skills employee given in resume")
  education:str=Field(description="Education employee given in resume")
  recent_project_title:str=Field(description="Recent Project employee given in resume")
  recent_project_details:str=Field(description="Recent Project Details employee given in resume")
  recent_project_tech_tools:List=Field(description="Recent Project Tech Tools employee given in resume")
  programming_languages:List=Field(description="Software languages employee given in resume")
  tools_and_technologies:Optional[List]=Field(description="Tools and technologies employee given in resume")
  soft_skills:List=Field(description="Soft Skills employee given in resume")
  certifications:List=Field(description="Certifications employee given in resume")
  work_location:str=Field(description="Current employee working location")
  unique_skills:str=Field(description="Unique Skills employee given in resume")
  work_experience:str=Field(description="Work Experience employee given in resume")
  candidate_name:str=Field(description="Candiate Name employee given in resume")
  strengths:str=Field(description="Strength of employee based on your analysis")
  weaknesses:str=Field(description="Weaknesses of employee based on your analysis")

class ParseJobDescription(BaseModel):
  job_title:Optional[str]=Field(description="current job title by company")
  years_of_experience:Optional[str]= Field(description="total years of relevant experience needed")
  industry:Optional[str]=Field(description="current industry which company looking for or industry company given")
  technical_skills:List=Field(description="Technical Skills company given")
  education:Optional[str]=Field(description="Education degree company given")
  programming_languages:List=Field(description="Software languages company given")
  tools_and_technologies:List=Field(description="Tools and technologies company given")
  soft_skills:List=Field(description="Soft Skills company given")
  responsibilities:Optional[str]=Field(description="responsibilities of given role")
  company_name:Optional[str]=Field(description="company name")
  job_location:Optional[str]=Field(description="job location")


def parse_input(state:Agent):

  llm=ChatGroq(model=state.get('llm_model')).with_structured_output(ParseResume)

  prompt ="""You are a parser who will be parsing user resume into structured prompt.
  Do your analysis based on resume and give employee strength and weakness based on your analysis. Use semantic skill mapping. Dont exactly look for key words for providing output
  Try to find all neccessary information if some informations are mismatching .Dont try to makeup any output.
  CRITICAL: Your entire response must be a single, valid JSON object. Do not include any text outside the JSON. Do not wrap it in code blocks or additional objects.
     """

  output=llm.invoke([("system",prompt),('user',state.get('raw_resume'))])

  llm=ChatGroq(model=state.get('llm_model')).with_structured_output(ParseJobDescription)

  prompt ="""You are a parser who will be parsing job description into structured prompt.
  Use semantic skill mapping. Dont exactly look for key words for providing output
  Do your analysis based on job description
  CRITICAL: Your entire response must be a single, valid JSON object. Do not include any text outside the JSON. Do not wrap it in code blocks or additional objects.
    """

  output1=llm.invoke([("system",prompt),('user',state.get('raw_job_description'))])

  return {'parsed_resume':{'job_title':output.job_title,
                           'years_of_experience':output.years_of_experience,
                           'industry':output.industry,
                           'no_of_projects':output.no_of_projects,
                           'technical_skills':output.technical_skills,
                           'education':output.education,
                           'recent_project_title':output.recent_project_title,
                           'recent_project_details':output.recent_project_details,
                           'recent_project_tech_tools':output.recent_project_tech_tools,
                           'programming_languages':output.programming_languages,
                           'tools_and_technologies':output.tools_and_technologies,
                           'soft_skills':output.soft_skills,
                           'certifications':output.certifications,
                           'work_location':output.work_location,
                           'unique_skills':output.unique_skills,
                           'work_experience':output.work_experience,
                           'candidate_name':output.candidate_name,
                           'strengths':output.strengths,
                           'weaknesses':output.weaknesses},
          'parsed_job_description': {'job_title':output1.job_title,
                                     'job_location':output1.job_location,
                                     'years_of_experience':output1.years_of_experience,
                                     'industry':output1.industry,
                                     'technical_skills':output1.technical_skills,
                                     'education':output1.education,
                                     'programming_languages':output1.programming_languages,
                                     'tools_and_technologies':output1.tools_and_technologies,'soft_skills':output1.soft_skills,
                                     'responsibilities':output1.responsibilities,'company_name':output1.company_name }}


def skills_assesment_agent(state:Agent):

  llm=ChatGroq(model=state.get('llm_model'))

  prompt="""You are an skill assesment agent for given job description for given user resume skills.
            Analyse user technical skills ,work experience,
            Semantic skill matching (React ↔ JavaScript, AWS ↔ Cloud)
            Identify transferable skills
            Analyse skill gaps between job description and user resume
            Check how user's strength can be eased to make fit for job description
            """

  output=llm.invoke([("system",prompt),('user',state.get('raw_resume')),('user',state.get('raw_job_description'))])

  return {'skill_assesment':output}

def experience_assesment_agent(state:Agent):

  llm=ChatGroq(model=state.get('llm_model'))

  prompt="""You are an experience assementer who will be identifying experience level and appropriateness
            Enssure experience to job description's requirements.
            Detect overqualification of user
            Detect if job description is over expectation or lot of skillsets.
            Analyse future carier growth in that role.
            Identify any red flag in given job description """
  output=llm.invoke([("system",prompt),('user',state.get('raw_resume')),('user',state.get('raw_job_description'))])

  return {'experience_assesment':output}

def decision_maker(state:Agent):

  llm=ChatGroq(model=state.get('llm_model'))

  prompt="""You are decision maker who will be aggregating output of various agents
            Duties:
            Analyse outputs of other agents.
            Generate reasoning and insights to user based on analysis of their job description and resume skills
            Give out recommendations to focus on weak areas
            Rate user' fit to job description in scale of 10
            Point out overqualification and any red flags in job description
            Guide user whether to apply or enhance resume or avoid applying it.
            If user is not fit to job description, explain why.
            If you want to enhance user resume, give out modified resume which can be applied.
            Maintain professionalism and concise.

            """
  output=llm.invoke([('system',prompt),
                     ('user',f"skill_assesment {state.get('skill_assesment')} and experience assesment is {state.get('experience_assesment')}")])

  return {'final_decision':output}

app=StateGraph(Agent)
app.add_node('document_intellligence',parse_input)
app.add_node('skill_assesment',skills_assesment_agent)
app.add_node('experience_assesment',experience_assesment_agent)
app.add_node('decision_maker',decision_maker)

app.add_edge(START,'document_intellligence')
app.add_edge('document_intellligence','skill_assesment')
app.add_edge('document_intellligence','experience_assesment')
app.add_edge('skill_assesment','decision_maker')
app.add_edge('experience_assesment','decision_maker')

model=app.compile()