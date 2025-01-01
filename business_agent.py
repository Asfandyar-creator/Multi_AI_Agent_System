from typing import TypedDict, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
import PyPDF2  # Updated import for PDF extraction

# Define the state
class State(TypedDict):
    problem_statement: Optional[str]
    analytical_problem: Optional[str]
    solution_attempt: Optional[str]
    final_solution: Optional[str]
    validation_requested: bool
    validation_feedback: Optional[str]
    evaluation_result: Optional[str]
    grade: Optional[str]
    progress: Optional[int]  # Track student progress

# Initialize GPT model
gpt = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Graph class implementation
class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, from_node, to_node):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)

    def add_conditional_edge(self, from_node, condition_func):
        self.conditional_edges[from_node] = condition_func

    def set_entry_point(self, name):
        self.entry_point = name

    def compile(self):
        def executor(state):
            current_node = self.entry_point
            while current_node:
                func = self.nodes.get(current_node)
                if func:
                    state = func(state)
                # Check if there's a conditional edge from this node
                if current_node in self.conditional_edges:
                    condition_func = self.conditional_edges[current_node]
                    current_node = condition_func(state)
                else:
                    # Follow the regular edge
                    next_nodes = self.edges.get(current_node, [])
                    if next_nodes:
                        current_node = next_nodes[0]
                    else:
                        break
            return state
        return executor

# Business Agent Functions
def provide_problem(state: State) -> State:
    # Path to the case study PDF
    pdf_path = "Tancent Monetization Case.pdf"
    
    # Extract the problem statement from the PDF
    state["problem_statement"] = extract_text_from_pdf(pdf_path)
    print("Business Agent: Problem statement extracted from the PDF and provided to the Student Agent.")
    return state

def evaluate_solution(state: State) -> State:
    # Simulate evaluating the final solution
    if state.get("final_solution"):
        state["evaluation_result"] = "Solution meets all requirements."
    else:
        state["evaluation_result"] = "Solution is incomplete."
    print(f"Business Agent: Solution evaluated. Result: {state['evaluation_result']}")
    return state

# Student Agent Functions
def receive_problem(state: State) -> State:
    # Receive the problem statement from the Business Agent
    print("Student Agent: Problem statement received from the Business Agent.")
    return state

def analyze_problem(state: State) -> State:
    # Convert the problem into an analytical format using GPT
    prompt = f"Convert the following problem into an analytical format:\n{state['problem_statement']}"
    response = gpt([HumanMessage(content=prompt)])
    state["analytical_problem"] = response.content
    print("Student Agent: Problem converted into an analytical format.")
    return state

def attempt_solution(state: State) -> State:
    # Attempt to solve the problem using GPT
    prompt = f"Solve the following problem:\n{state['analytical_problem']}"
    response = gpt([HumanMessage(content=prompt)])
    state["solution_attempt"] = response.content
    print("Student Agent: Solution attempted.")
    return state

def request_guidance(state: State) -> State:
    # Simulate requesting guidance from the Teacher or TA Agent
    if "confusion" in state["solution_attempt"].lower():
        print("Student Agent: Requesting guidance from the Teacher Agent.")
    elif "specific_task_help" in state["solution_attempt"].lower():
        print("Student Agent: Requesting task-specific guidance from the TA Agent.")
    else:
        print("Student Agent: No guidance needed. Proceeding to finalize the solution.")
    return state

def finalize_solution(state: State) -> State:
    # Finalize the solution
    state["final_solution"] = state["solution_attempt"]
    print("Student Agent: Solution finalized.")
    return state

# Teacher Agent Functions
def store_solution_knowledgebase(state: State) -> State:
    # Path to the solution PDF
    pdf_path = "Tancent TN.pdf"
    
    # Extract the solution text from the PDF
    solution_text = extract_text_from_pdf(pdf_path)
    
    # Store the solution text as embeddings in FAISS
    global knowledge_base  # Use a global variable for the knowledge base
    knowledge_base = FAISS.from_texts([solution_text], embeddings)
    print("Teacher Agent: Solution PDF stored as embeddings in the knowledge base.")
    return state

def provide_broad_guidance(state: State) -> State:
    # Provide broad guidance using GPT
    prompt = f"The student is stuck on this problem:\n{state['problem_statement']}\nProvide broad guidance."
    response = gpt([HumanMessage(content=prompt)])
    state["solution_attempt"] += f"\nTeacher's Guidance: {response.content}"
    print("Teacher Agent: Broad guidance provided to the Student Agent.")
    return state

def validate_solution(state: State) -> State:
    # Validate the student's solution using the knowledge base
    if state.get("final_solution"):
        # Search the knowledge base for similar content
        docs = knowledge_base.similarity_search(state["final_solution"], k=1)
        if docs:
            state["validation_feedback"] = f"Solution is valid. Reference: {docs[0].page_content}"
        else:
            state["validation_feedback"] = "Solution is invalid. No matching reference found."
    else:
        state["validation_feedback"] = "No solution provided for validation."
    print(f"Teacher Agent: Solution validated. Feedback: {state['validation_feedback']}")
    return state

def grade_student_teacher(state: State) -> State:
    if not state.get("solution_attempt"):
        state["grade"] = "Incomplete"
    elif "confusion" in state["solution_attempt"].lower():
        state["grade"] = "B"
    else:
        state["grade"] = "A"
    print(f"Teacher Agent: Student graded. Grade: {state['grade']}")
    return state

# TA Agent Functions
def provide_task_specific_guidance(state: State) -> State:
    # Provide task-specific guidance using GPT
    prompt = f"The student needs help with this specific task:\n{state['solution_attempt']}\nProvide task-specific guidance."
    response = gpt([HumanMessage(content=prompt)])
    state["solution_attempt"] += f"\nTA's Guidance: {response.content}"
    print("TA Agent: Task-specific guidance provided to the Student Agent.")
    return state

def track_progress(state: State) -> State:
    # Track the student's progress
    state["progress"] = 100  # Simulate 100% progress
    print(f"TA Agent: Student progress tracked. Progress: {state['progress']}%")
    return state

def grade_student_progress(state: State) -> State:
    if not state.get("progress"):
        state["grade"] = "Incomplete"
    elif state["progress"] >= 80:
        state["grade"] = "A"
    else:
        state["grade"] = "B"
    print(f"TA Agent: Student graded based on progress. Grade: {state['grade']}")
    return state

# Create the graph for the Business Agent
business_workflow = Graph()
business_workflow.add_node("provide_problem", provide_problem)
business_workflow.add_node("evaluate_solution", evaluate_solution)
business_workflow.add_edge("provide_problem", "evaluate_solution")
business_workflow.set_entry_point("provide_problem")
business_app = business_workflow.compile()

# Create the graph for the Student Agent
student_workflow = Graph()
student_workflow.add_node("receive_problem", receive_problem)
student_workflow.add_node("analyze_problem", analyze_problem)
student_workflow.add_node("attempt_solution", attempt_solution)
student_workflow.add_node("request_guidance", request_guidance)
student_workflow.add_node("finalize_solution", finalize_solution)
student_workflow.add_edge("receive_problem", "analyze_problem")
student_workflow.add_edge("analyze_problem", "attempt_solution")
student_workflow.add_conditional_edge("attempt_solution", lambda state: "request_guidance" if "confusion" in state["solution_attempt"].lower() or "specific_task_help" in state["solution_attempt"].lower() else "finalize_solution")
student_workflow.set_entry_point("receive_problem")
student_app = student_workflow.compile()

# Create the graph for the Teacher Agent
teacher_workflow = Graph()
teacher_workflow.add_node("store_solution_knowledgebase", store_solution_knowledgebase)
teacher_workflow.add_node("provide_broad_guidance", provide_broad_guidance)
teacher_workflow.add_node("validate_solution", validate_solution)
teacher_workflow.add_node("grade_student_teacher", grade_student_teacher)
teacher_workflow.add_edge("store_solution_knowledgebase", "provide_broad_guidance")
teacher_workflow.add_edge("provide_broad_guidance", "validate_solution")
teacher_workflow.add_edge("validate_solution", "grade_student_teacher")
teacher_workflow.set_entry_point("store_solution_knowledgebase")
teacher_app = teacher_workflow.compile()

# Create the graph for the TA Agent
ta_workflow = Graph()
ta_workflow.add_node("provide_task_specific_guidance", provide_task_specific_guidance)
ta_workflow.add_node("track_progress", track_progress)
ta_workflow.add_node("grade_student_progress", grade_student_progress)
ta_workflow.add_edge("provide_task_specific_guidance", "track_progress")
ta_workflow.add_edge("track_progress", "grade_student_progress")
ta_workflow.set_entry_point("provide_task_specific_guidance")
ta_app = ta_workflow.compile()

# Initialize knowledge_base as a global variable
knowledge_base = None

# Run the Business Agent workflow
business_initial_state = State(
    problem_statement=None,
    analytical_problem=None,
    solution_attempt=None,
    final_solution=None,
    validation_requested=False,
    validation_feedback=None,
    evaluation_result=None,
    grade=None,
    progress=None
)
business_final_state = business_app.invoke(business_initial_state)

# Pass the problem statement to the Student Agent
student_initial_state = State(
    problem_statement=business_final_state.get("problem_statement"),
    analytical_problem=None,
    solution_attempt=None,
    final_solution=None,
    validation_requested=False,
    validation_feedback=None,
    evaluation_result=None,
    grade=None,
    progress=None
)

# Run the Student Agent workflow
student_final_state = student_app.invoke(student_initial_state)

# Pass the Student Agent's state to the Teacher Agent
teacher_initial_state = State(
    problem_statement=student_final_state.get("problem_statement"),
    analytical_problem=student_final_state.get("analytical_problem"),
    solution_attempt=student_final_state.get("solution_attempt"),
    final_solution=student_final_state.get("final_solution"),
    validation_requested=False,
    validation_feedback=None,
    evaluation_result=None,
    grade=None,
    progress=None
)

# Run the Teacher Agent workflow
teacher_final_state = teacher_app.invoke(teacher_initial_state)

# Pass the Student Agent's state to the TA Agent
ta_initial_state = State(
    problem_statement=student_final_state.get("problem_statement"),
    analytical_problem=student_final_state.get("analytical_problem"),
    solution_attempt=student_final_state.get("solution_attempt"),
    final_solution=student_final_state.get("final_solution"),
    validation_requested=False,
    validation_feedback=None,
    evaluation_result=None,
    grade=None,
    progress=None
)

# Run the TA Agent workflow
ta_final_state = ta_app.invoke(ta_initial_state)

# Pass the final solution back to the Business Agent for evaluation
business_final_state["final_solution"] = student_final_state.get("final_solution")
business_final_state = business_app.invoke(business_final_state)

# Print final results
print("\nBusiness Agent Final State:", business_final_state)
print("Student Agent Final State:", student_final_state)
print("Teacher Agent Final State:", teacher_final_state)
print("TA Agent Final State:", ta_final_state)