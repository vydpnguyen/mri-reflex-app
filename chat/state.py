import reflex as rx
import google.generativeai as genai

# Define the API key directly in the script
API_KEY = "AIzaSyDGY_bjuq_LwQOfbo1rhFuvCLstU9Fv51E"
genai.configure(api_key=API_KEY)

class QA(rx.Base):
    """A question and answer pair."""
    question: str
    answer: str

DEFAULT_CHATS = {
    "Intros": [],
}

class State(rx.State):
    """The app state."""
    chats: dict[str, list[QA]] = DEFAULT_CHATS
    current_chat = "Intros"
    processing: bool = False
    new_chat_name: str = ""

    def create_chat(self):
        """Create a new chat."""
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat."""
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles."""
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        question = form_data["question"]
        if question == "":
            return

        async for value in self.generate_response(question):
            yield value

    async def generate_response(self, question: str):
        """Process the question using Google's generative AI API."""
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)
        self.processing = True
        yield

        full_prompt = ("You are now an AI chatbot assistant designed to provide suggestions for courses of action based on "
        "medical diagnoses. You do not create or confirm diagnoses but rather support medical professionals by "
        "suggesting possible actions that can be taken after a diagnosis has been made by a qualified individual. "
        "You must comply with all healthcare regulations and privacy laws, and your guidance should always suggest "
        "consulting with a healthcare provider. Given this context, please respond to the following inquiry: " + question)
    
        # Instantiate the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the content
        response = model.generate_content(full_prompt)
    
        
        # Retrieve the text from the response
        answer_text = response.text if response.text else ""
        self.chats[self.current_chat][-1].answer += answer_text
        self.chats = self.chats
        yield

        self.processing = False
