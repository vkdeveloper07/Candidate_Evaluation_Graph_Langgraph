import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


class EvaluationInput(BaseModel):
    candidate_id: str
    candidate_resume: str
    job_description: str

class EvaluationState(TypedDict, total=False):
    candidate_id: str
    candidate_resume: str
    job_description: str

    technical_evaluation: str
    experience_evaluation: str
    profile_clarity_evaluation: str

    candidate_fit: Literal[
        "potential_fit",
        "substantial_preparation"
    ]

    final_output: str


class CandidateFitDecision(BaseModel):
    candidate_fit: Literal[
        "potential_fit",
        "substantial_preparation"
    ]


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

decision_llm = llm.with_structured_output(CandidateFitDecision)


def evaluate_technical_skills(state: EvaluationState) -> dict:
    """Compare the candidate's technical skills with the job requirements."""

    response = llm.invoke(
        f"""
        Compare the technical skills required by the job with the
        skills demonstrated in the candidate's resume.

        Mention:
        - Matching skills
        - Missing skills
        - Evidence from the resume

        Resume:
        {state["candidate_resume"]}

        Job Description:
        {state["job_description"]}
        """
    )

    return {
        "technical_evaluation": response.content
    }


def evaluate_experience_fit(state: EvaluationState) -> dict:
    """Check whether the candidate's experience matches the role."""

    response = llm.invoke(
        f"""
        Compare the candidate's experience with the job description.

        Consider:
        - Responsibilities
        - Seniority
        - Domain experience
        - Measurable outcomes
        - Experience gaps

        Explain the reasoning using evidence from the resume.

        Resume:
        {state["candidate_resume"]}

        Job Description:
        {state["job_description"]}
        """
    )

    return {
        "experience_evaluation": response.content
    }


def evaluate_profile_clarity(state: EvaluationState) -> dict:
    """Review how clearly the resume communicates the candidate's experience."""

    response = llm.invoke(
        f"""
        Review the candidate's resume from a recruiter perspective.

        Check:
        - Clarity
        - Structure
        - Evidence of achievements
        - Measurable impact
        - Unclear or unsupported claims
        - Relevance to the job

        Give practical suggestions for improvement.

        Resume:
        {state["candidate_resume"]}

        Job Description:
        {state["job_description"]}
        """
    )

    return {
        "profile_clarity_evaluation": response.content
    }


def classify_candidate_fit(state: EvaluationState) -> dict:
    """Decide whether the candidate is a potential fit."""

    response = decision_llm.invoke(
        f"""
        Based on the three evaluations below, classify the candidate
        as either:

        1. potential_fit
        2. substantial_preparation

        Technical evaluation:
        {state["technical_evaluation"]}

        Experience evaluation:
        {state["experience_evaluation"]}

        Profile clarity evaluation:
        {state["profile_clarity_evaluation"]}
        """
    )

    return {
        "candidate_fit": response.candidate_fit
    }


def interview_preparation_pack(state: EvaluationState) -> dict:
    """Prepare an interview plan for a potential-fit candidate."""

    response = llm.invoke(
        f"""
        Create a practical interview preparation plan for this candidate.

        Include:
        - Likely technical questions
        - Likely behavioral questions
        - Resume points the candidate should be ready to explain
        - Areas the interviewer may challenge
        - Preparation actions

        Resume:
        {state["candidate_resume"]}

        Job Description:
        {state["job_description"]}

        Technical evaluation:
        {state["technical_evaluation"]}

        Experience evaluation:
        {state["experience_evaluation"]}

        Profile clarity evaluation:
        {state["profile_clarity_evaluation"]}
        """
    )

    return {
        "final_output": response.content
    }


def candidate_gap_plan(state: EvaluationState) -> dict:
    """Create a preparation plan for a candidate with significant gaps."""

    response = llm.invoke(
        f"""
        Create a practical gap and preparation plan for this candidate.

        Include:
        - Missing technical skills
        - Experience gaps
        - Missing evidence in the resume
        - Resume improvements
        - Communication improvements
        - Concrete next steps

        Resume:
        {state["candidate_resume"]}

        Job Description:
        {state["job_description"]}

        Technical evaluation:
        {state["technical_evaluation"]}

        Experience evaluation:
        {state["experience_evaluation"]}

        Profile clarity evaluation:
        {state["profile_clarity_evaluation"]}
        """
    )

    return {
        "final_output": response.content
    }


def route_candidate(state: EvaluationState) -> str:
    """Choose the final node based on the candidate classification."""

    if state["candidate_fit"] == "potential_fit":
        return "interview_preparation_pack"

    return "candidate_gap_plan"


def build_graph():
    graph = StateGraph(EvaluationState)

    # Three independent evaluations
    graph.add_node("evaluate_technical_skills", evaluate_technical_skills)
    graph.add_node("evaluate_experience_fit", evaluate_experience_fit)
    graph.add_node("evaluate_profile_clarity", evaluate_profile_clarity)

    # Decision node
    graph.add_node("classify_candidate_fit", classify_candidate_fit)

    # Final nodes
    graph.add_node("interview_preparation_pack", interview_preparation_pack)
    graph.add_node("candidate_gap_plan", candidate_gap_plan)

    # Start all three evaluations independently
    graph.add_edge(START, "evaluate_technical_skills")
    graph.add_edge(START, "evaluate_experience_fit")
    graph.add_edge(START, "evaluate_profile_clarity")

    # Wait for all three evaluations before making the decision
    graph.add_edge(
        [
            "evaluate_technical_skills",
            "evaluate_experience_fit",
            "evaluate_profile_clarity",
        ],
        "classify_candidate_fit",
    )

    # Route based on the decision
    graph.add_conditional_edges(
        "classify_candidate_fit",
        route_candidate,
        {
            "interview_preparation_pack": "interview_preparation_pack",
            "candidate_gap_plan": "candidate_gap_plan",
        },
    )

    # Both final paths end the graph
    graph.add_edge("interview_preparation_pack", END)
    graph.add_edge("candidate_gap_plan", END)

    return graph.compile()


candidate_evaluation_graph = build_graph()


def evaluate_candidate(evaluation_input: EvaluationInput) -> EvaluationState:
    """Run the candidate evaluation workflow."""

    return candidate_evaluation_graph.invoke(
        evaluation_input.model_dump()
    )

if __name__ == "__main__":

    evaluation_input = EvaluationInput(
        candidate_id="C001",
        candidate_resume="""
        11 years of experience in software development.
        Strong experience in C#, .NET, ASP.NET Core, Vue.js, Azure,
        REST APIs, microservices and SQL.

        Currently working as a Senior Software Engineer.
        Experience building enterprise applications and BFF APIs.
        """,
        job_description="""
        We are looking for a Senior .NET Software Engineer.

        Requirements:
        - 8+ years of software development experience
        - Strong C# and .NET experience
        - ASP.NET Core
        - REST APIs
        - Microservices
        - Azure
        - SQL
        - Experience with system design
        """
    )

    result = evaluate_candidate(evaluation_input)

    print("\nCandidate ID:", result["candidate_id"])
    print("\nCandidate Fit:", result["candidate_fit"])

    print("\n--- Technical Evaluation ---")
    print(result["technical_evaluation"])

    print("\n--- Experience Evaluation ---")
    print(result["experience_evaluation"])

    print("\n--- Profile Clarity Evaluation ---")
    print(result["profile_clarity_evaluation"])

    print("\n--- Final Output ---")
    print(result["final_output"])